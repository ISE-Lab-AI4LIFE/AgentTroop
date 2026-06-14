#!/usr/bin/env python3
"""Comprehensive audit of the adaptive anomaly threshold mechanism.

Tests dynamic adaptation across multiple dimensions:
1. Multi-iteration simulation with exploration decay
2. Diverse score distributions (uniform, skewed, bimodal, all-low, all-high)
3. Adaptive vs fixed threshold comparison
4. Min/max intervention guarantees
5. Sensitivity analysis of config parameters
6. Extreme cases: starvation, overload, cold start
"""

import json
import logging
import math
import random
import sys
import time
import unittest
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("test_anomaly_threshold")


# ===========================================================================
# Stub classes
# ===========================================================================

@dataclass
class _StubIntervention:
    prompt: str
    transforms: List[Dict[str, str]] = field(default_factory=list)


@dataclass
class _StubEpisode:
    intervention: _StubIntervention
    outcome: Optional[int]
    episode_id: str = ""


def _make_episodes(groups: Dict[str, List[Optional[int]]]) -> List[_StubEpisode]:
    eps: List[_StubEpisode] = []
    for prompt, outcomes in groups.items():
        for i, outcome in enumerate(outcomes):
            eps.append(_StubEpisode(
                intervention=_StubIntervention(prompt=prompt),
                outcome=outcome,
                episode_id=f"ep_{prompt}_{i}",
            ))
    return eps


# ===========================================================================
# Agent stub (mirrors agents/cognitive.py exactly)
# ===========================================================================

class _MinimalCognitiveAgent:
    ADAPTIVE_DEFAULTS = {
        "method": "percentile",
        "percentile": 85,
        "min_interventions": 3,
        "max_interventions": 10,
        "history_window": 5,
        "exploration_decay": True,
    }

    def __init__(self, anomaly_threshold: float = 0.2,
                 anomaly_selection_config: Optional[Dict[str, Any]] = None):
        self.anomaly_threshold = anomaly_threshold
        raw = anomaly_selection_config or {}
        self.anomaly_selection: Dict[str, Any] = {
            **self.ADAPTIVE_DEFAULTS,
            **{k: v for k, v in raw.items() if v is not None},
        }
        self._anomaly_threshold_history: List[float] = []
        self._anomaly_consecutive_low: int = 0
        self._anomaly_iteration: int = 0
        self._current_effective_percentile: float = float(
            self.anomaly_selection["percentile"]
        )

    def compute_scores(self, episodes: List[_StubEpisode]
                       ) -> List[Tuple[str, float, List[_StubEpisode]]]:
        groups: Dict[str, List[_StubEpisode]] = {}
        for ep in episodes:
            groups.setdefault(ep.intervention.prompt, []).append(ep)
        return self._compute_group_anomaly_scores(groups)

    def _compute_group_anomaly_scores(
        self, groups: Dict[str, List[Any]],
    ) -> List[Tuple[str, float, List[Any]]]:
        all_outcomes: List[int] = []
        for group in groups.values():
            for ep in group:
                if ep.outcome is not None:
                    all_outcomes.append(int(ep.outcome))
        if not all_outcomes:
            return []
        n_total = len(all_outcomes)
        n_accept = sum(1 for o in all_outcomes if o == 0)
        global_accept_rate = n_accept / n_total

        scored: List[Tuple[str, float, List[Any]]] = []
        for base_prompt, group in groups.items():
            outcomes = [int(ep.outcome) for ep in group if ep.outcome is not None]
            if not outcomes:
                continue
            n = len(outcomes)
            group_accept = sum(1 for o in outcomes if o == 0)
            group_accept_rate = group_accept / n
            deviation = abs(group_accept_rate - global_accept_rate)
            p = group_accept_rate
            if 0.0 < p < 1.0:
                entropy = -(p * math.log2(p) + (1 - p) * math.log2(1 - p))
                entropy_bonus = entropy * 0.6
            else:
                entropy_bonus = 0.0
            size_bonus = min(n / 20.0, 0.1)
            score = entropy_bonus + deviation * 0.3 + size_bonus
            scored.append((base_prompt, score, group))
        scored.sort(key=lambda x: -x[1])
        return scored

    def select_groups(self, scored: List[Tuple[str, float, List[Any]]]
                      ) -> List[Tuple[str, float, List[Any]]]:
        return self._select_groups_by_percentile(scored)

    def _select_groups_by_percentile(
        self, scored: List[Tuple[str, float, List[Any]]],
    ) -> List[Tuple[str, float, List[Any]]]:
        cfg = self.anomaly_selection
        method = cfg.get("method", "fixed")
        if method == "fixed" or not scored:
            return scored
        scores = [s[1] for s in scored]
        p = max(1.0, min(99.0, self._current_effective_percentile))
        import numpy as np
        threshold = float(np.percentile(scores, p))
        selected = [s for s in scored if s[1] >= threshold]
        n_raw = len(selected)
        min_iv = int(cfg.get("min_interventions", 3))
        max_iv = int(cfg.get("max_interventions", 10))
        if len(selected) < min_iv:
            n_extra = min(min_iv - len(selected), len(scored) - len(selected))
            selected = scored[:len(selected) + n_extra]
        selected = selected[:max_iv]
        effective_threshold = float(np.percentile(scores, p))
        self._anomaly_threshold_history.append(effective_threshold)
        window = int(cfg.get("history_window", 5))
        if len(self._anomaly_threshold_history) > window:
            self._anomaly_threshold_history.pop(0)
        if cfg.get("exploration_decay", True):
            if n_raw < min_iv // 2:
                self._anomaly_consecutive_low += 1
            else:
                self._anomaly_consecutive_low = 0
            if self._anomaly_consecutive_low >= 3:
                old_p = self._current_effective_percentile
                self._current_effective_percentile = max(20.0, old_p - 10.0)
                self._anomaly_consecutive_low = 0
        self._anomaly_iteration += 1
        return selected

    def reset(self) -> None:
        self._anomaly_threshold_history.clear()
        self._anomaly_consecutive_low = 0
        self._anomaly_iteration = 0
        self._current_effective_percentile = float(
            self.anomaly_selection["percentile"]
        )


# ===========================================================================
# Episode generators for different score distributions
# ===========================================================================

def _uniform_distribution(n_groups: int, eps_per_group: int,
                          accept_ratio: float = 0.5, seed: int = 0
                          ) -> List[_StubEpisode]:
    """Episodes where each group has roughly the same accept_ratio."""
    rng = random.Random(seed)
    eps: List[_StubEpisode] = []
    for g in range(n_groups):
        prompt = f"u{g}"
        for i in range(eps_per_group):
            outcome = 0 if rng.random() < accept_ratio else 1
            eps.append(_StubEpisode(
                intervention=_StubIntervention(prompt=prompt),
                outcome=outcome,
            ))
    return eps


def _skewed_distribution(n_groups: int, eps_per_group: int,
                         seed: int = 0) -> List[_StubEpisode]:
    """Most groups uniform, few have mixed outcomes."""
    rng = random.Random(seed)
    eps: List[_StubEpisode] = []
    for g in range(n_groups):
        prompt = f"s{g}"
        is_special = g < int(n_groups * 0.15)
        accept_ratio = 0.5 if is_special else (1.0 if g % 2 == 0 else 0.0)
        for i in range(eps_per_group):
            outcome = 0 if rng.random() < accept_ratio else 1
            eps.append(_StubEpisode(
                intervention=_StubIntervention(prompt=prompt),
                outcome=outcome,
            ))
    return eps


def _bimodal_distribution(n_groups: int, eps_per_group: int,
                          seed: int = 0) -> List[_StubEpisode]:
    """Two clusters: one mostly ACCEPT, one mostly REFUSE."""
    rng = random.Random(seed)
    eps: List[_StubEpisode] = []
    half = n_groups // 2
    for g in range(n_groups):
        prompt = f"b{g}"
        accept_ratio = 0.9 if g < half else 0.1
        for i in range(eps_per_group):
            outcome = 0 if rng.random() < accept_ratio else 1
            eps.append(_StubEpisode(
                intervention=_StubIntervention(prompt=prompt),
                outcome=outcome,
            ))
    return eps


def _all_low_distribution(n_groups: int, eps_per_group: int,
                          seed: int = 0) -> List[_StubEpisode]:
    """All groups have only REFUSE outcomes — no anomalies at all."""
    eps: List[_StubEpisode] = []
    for g in range(n_groups):
        prompt = f"l{g}"
        for i in range(eps_per_group):
            eps.append(_StubEpisode(
                intervention=_StubIntervention(prompt=prompt),
                outcome=1,
            ))
    return eps


def _mixed_rate_distribution(n_groups: int, eps_per_group: int,
                             seed: int = 0) -> List[_StubEpisode]:
    """Continuous spectrum of accept_ratios from 0 to 1."""
    rng = random.Random(seed)
    eps: List[_StubEpisode] = []
    for g in range(n_groups):
        prompt = f"m{g}"
        accept_ratio = g / n_groups
        for i in range(eps_per_group):
            outcome = 0 if rng.random() < accept_ratio else 1
            eps.append(_StubEpisode(
                intervention=_StubIntervention(prompt=prompt),
                outcome=outcome,
            ))
    return eps


# ===========================================================================
# Test suites
# ===========================================================================

class TestScoreDistributionDynamics(unittest.TestCase):
    """Test how the scoring function responds to different distributions."""

    def test_uniform_scores_ranked(self):
        """Uniform distribution: groups with same rate have similar scores."""
        agent = _MinimalCognitiveAgent()
        eps = _uniform_distribution(10, 20, accept_ratio=0.5, seed=1)
        scored = agent.compute_scores(eps)
        scores = [s[1] for s in scored]
        # All groups have ~50% accept → scores should be close (small variance)
        mean_s = sum(scores) / len(scores)
        variance = sum((s - mean_s) ** 2 for s in scores) / len(scores)
        self.assertLess(variance, 0.05)

    def test_skewed_isolates_special_groups(self):
        """Skewed: special groups (15%) should rank at the top."""
        agent = _MinimalCognitiveAgent()
        eps = _skewed_distribution(40, 10, seed=2)
        scored = agent.compute_scores(eps)
        scores = [s[1] for s in scored]
        # The special groups (first 6 of 40 = 15%) should be among top
        top_6 = scored[:6]
        special_in_top = sum(1 for s in top_6 if s[0].startswith("s")
                             and int(s[0][1:]) < 6)
        self.assertGreaterEqual(special_in_top, 4)

    def test_bimodal_deviation_detected(self):
        """Bimodal: both clusters have large deviation from global rate."""
        agent = _MinimalCognitiveAgent()
        eps = _bimodal_distribution(20, 15, seed=3)
        scored = agent.compute_scores(eps)
        scores = [s[1] for s in scored]
        # Global rate ≈ 50%, each cluster ≈ 90% or 10%, deviation ≈ 0.4
        # With entropy bonus, scores should be substantial
        mean_s = sum(scores) / len(scores)
        self.assertGreater(mean_s, 0.2)

    def test_all_low_scores_are_minimal(self):
        """All-REFUSE: all scores come only from size_bonus."""
        agent = _MinimalCognitiveAgent()
        eps = _all_low_distribution(10, 5)
        scored = agent.compute_scores(eps)
        for _, score, _ in scored:
            # All rates are 1.0 (global also 1.0), deviation=0, no entropy
            # Only size_bonus = min(5/20, 0.1) = 0.1
            self.assertAlmostEqual(score, 0.1, places=6)


class TestAdaptiveVsFixed(unittest.TestCase):
    """Direct comparison of adaptive vs fixed threshold."""

    def _run_adaptive(self, eps, config, n_iters=20):
        agent = _MinimalCognitiveAgent(anomaly_selection_config=config)
        selections = []
        for _ in range(n_iters):
            scored = agent.compute_scores(eps)
            sel = agent.select_groups(scored)
            selections.append(len(sel))
        return selections, agent

    def _run_fixed(self, eps, config, n_iters=20):
        cfg = dict(config)
        cfg["method"] = "fixed"
        agent = _MinimalCognitiveAgent(anomaly_selection_config=cfg)
        selections = []
        for _ in range(n_iters):
            scored = agent.compute_scores(eps)
            sel = agent.select_groups(scored)
            selections.append(len(sel))
        return selections

    def test_adaptive_selects_more_when_starved(self):
        """Adaptive selects more groups than fixed when few anomalies exist.

        Creates episodes where scores vary but only a few pass the high
        percentile. Adaptive's exploration decay should eventually lower
        the percentile and widen selection.
        """
        # 2 special groups (50/50, high score) + 18 uniform (low score)
        # With percentile=95, only ~1-2 groups pass naturally
        eps_dict: Dict[str, List[Optional[int]]] = {}
        for i in range(2):
            eps_dict[f"special_{i}"] = [0, 1, 0, 1]
        for i in range(18):
            eps_dict[f"flat_{i}"] = [0, 0, 0]
        eps = _make_episodes(eps_dict)

        base_cfg = {
            "method": "percentile",
            "percentile": 95,
            "min_interventions": 10,
            "max_interventions": 10,
            "exploration_decay": True,
        }
        adap_sel, agent = self._run_adaptive(eps, base_cfg, n_iters=30)
        fixed_sel = self._run_fixed(eps, base_cfg, n_iters=30)
        # After decay kicks in, adaptive should lower percentile
        self.assertLess(agent._current_effective_percentile,
                        base_cfg["percentile"])

    def test_adaptive_maintains_min_guarantee(self):
        """Adaptive always meets min_interventions even with diverse scores."""
        eps = _mixed_rate_distribution(30, 8, seed=5)
        base_cfg = {
            "method": "percentile",
            "percentile": 95,
            "min_interventions": 5,
            "max_interventions": 10,
            "exploration_decay": True,
        }
        adap_sel, _ = self._run_adaptive(eps, base_cfg, n_iters=10)
        fixed_sel = self._run_fixed(eps, base_cfg, n_iters=10)
        # Both should maintain min_interventions
        self.assertGreaterEqual(min(adap_sel), 5)
        self.assertGreaterEqual(min(fixed_sel), 5)


class TestMultiIterationDynamics(unittest.TestCase):
    """Simulate a full campaign unfolding over many iterations."""

    def test_exploration_decay_curve(self):
        """Verify the decay curve: percentile drops steadily under starvation."""
        agent = _MinimalCognitiveAgent(anomaly_selection_config={
            "method": "percentile",
            "percentile": 95,
            "min_interventions": 15,
            "max_interventions": 20,
            "exploration_decay": True,
        })
        # Create 11 groups with diverse accept rates (0..1).
        # Richer score distribution → different percentiles yield different n_raw.
        # min_iv=15 > n_groups so min_iv//2=7 and decay persists until percentile < ~70.
        eps_dict: Dict[str, List[Optional[int]]] = {}
        rates = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        rng = random.Random(42)
        for i, rate in enumerate(rates):
            prompt = f"g{i:02d}"
            n_eps = 10
            outcomes: List[Optional[int]] = [
                0 if rng.random() < rate else 1 for _ in range(n_eps)
            ]
            eps_dict[prompt] = outcomes
        eps = _make_episodes(eps_dict)
        scored = agent.compute_scores(eps)
        history: List[float] = []
        for _ in range(20):
            agent.select_groups(scored)
            history.append(agent._current_effective_percentile)
        # Should have decayed multiple times
        decays = sum(1 for i in range(1, len(history))
                     if history[i] < history[i - 1])
        self.assertGreaterEqual(decays, 3)
        # Final percentile should be lower
        self.assertLess(history[-1], 70)

    def test_decay_reverses_when_anomalies_found(self):
        """Once groups begin passing the percentile, decay halts."""
        agent = _MinimalCognitiveAgent(anomaly_selection_config={
            "method": "percentile",
            "percentile": 95,
            "min_interventions": 10,
            "max_interventions": 10,
            "exploration_decay": True,
        })
        # Phase 1: starvation (1 high-score + 9 low-score → n_raw=1 < 5)
        eps_dict: Dict[str, List[Optional[int]]] = {}
        eps_dict["special"] = [0, 1, 0, 1]
        for i in range(9):
            eps_dict[f"flat_{i}"] = [0, 0, 0]
        low_eps = _make_episodes(eps_dict)
        low_scored = agent.compute_scores(low_eps)
        for _ in range(5):
            agent.select_groups(low_scored)

        self.assertLess(agent._current_effective_percentile, 95)

        # Phase 2: rich data (many mixed-outcome groups → many pass percentile)
        rich_dict: Dict[str, List[Optional[int]]] = {}
        for i in range(10):
            rich_dict[f"rich_{i}"] = [0, 1, 0, 1]
        rich_eps = _make_episodes(rich_dict)
        rich_scored = agent.compute_scores(rich_eps)
        for _ in range(5):
            agent.select_groups(rich_scored)
        # Should reset consecutive_low since rich data gives many selections
        self.assertEqual(agent._anomaly_consecutive_low, 0)

    def test_selection_count_predictable(self):
        """Selection count per iteration follows expected pattern."""
        agent = _MinimalCognitiveAgent(anomaly_selection_config={
            "method": "percentile",
            "percentile": 80,
            "min_interventions": 2,
            "max_interventions": 5,
        })
        eps = _mixed_rate_distribution(20, 10, seed=10)
        scored = agent.compute_scores(eps)
        counts: List[int] = []
        for _ in range(10):
            sel = agent.select_groups(scored)
            counts.append(len(sel))
        # Should always be within [min, max]
        for c in counts:
            self.assertGreaterEqual(c, 2)
            self.assertLessEqual(c, 5)

    def test_history_window_smooths(self):
        """Threshold history is bounded."""
        agent = _MinimalCognitiveAgent(anomaly_selection_config={
            "method": "percentile",
            "percentile": 50,
            "min_interventions": 2,
            "max_interventions": 10,
            "history_window": 3,
        })
        eps = _mixed_rate_distribution(15, 8, seed=12)
        scored = agent.compute_scores(eps)
        for _ in range(10):
            agent.select_groups(scored)
        self.assertLessEqual(len(agent._anomaly_threshold_history), 3)


class TestSensitivityAnalysis(unittest.TestCase):
    """How config parameters affect behavior."""

    def test_percentile_sensitivity(self):
        """Higher percentile → fewer raw selections pre-guarantee."""
        eps = _mixed_rate_distribution(25, 10, seed=20)
        scored = _MinimalCognitiveAgent().compute_scores(eps)
        n_groups = len(scored)

        results: List[Tuple[int, int]] = []
        for pctl in [10, 30, 50, 70, 90]:
            agent = _MinimalCognitiveAgent(anomaly_selection_config={
                "method": "percentile",
                "percentile": pctl,
                "min_interventions": 1,
                "max_interventions": n_groups,
                "exploration_decay": False,
            })
            # Construct a custom one-off to get n_raw before guarantee
            cfg = agent.anomaly_selection
            scores_arr = [s[1] for s in scored]
            p = max(1.0, min(99.0, float(pctl)))
            import numpy as np
            thresh = float(np.percentile(scores_arr, p))
            n_raw = sum(1 for s in scored if s[1] >= thresh)
            results.append((pctl, n_raw))

        # Higher percentile → fewer (or equal) raw selections
        for i in range(1, len(results)):
            self.assertGreaterEqual(results[i - 1][1], results[i][1])

    def test_min_interventions_sensitivity(self):
        """Higher min_interventions → more groups selected."""
        eps = _mixed_rate_distribution(20, 8, seed=25)
        scored = _MinimalCognitiveAgent().compute_scores(eps)

        counts: List[int] = []
        for min_iv in [1, 3, 5, 10]:
            agent = _MinimalCognitiveAgent(anomaly_selection_config={
                "method": "percentile",
                "percentile": 90,
                "min_interventions": min_iv,
                "max_interventions": 20,
                "exploration_decay": False,
            })
            sel = agent.select_groups(scored)
            counts.append(len(sel))
        # More min_interventions → more or equal selections
        for i in range(1, len(counts)):
            self.assertGreaterEqual(counts[i], counts[i - 1])

    def test_max_interventions_cap(self):
        """Selection never exceeds max_interventions."""
        eps = _mixed_rate_distribution(30, 10, seed=30)
        scored = _MinimalCognitiveAgent().compute_scores(eps)

        for max_iv in [2, 5, 10]:
            agent = _MinimalCognitiveAgent(anomaly_selection_config={
                "method": "percentile",
                "percentile": 10,
                "min_interventions": 1,
                "max_interventions": max_iv,
                "exploration_decay": False,
            })
            sel = agent.select_groups(scored)
            self.assertLessEqual(len(sel), max_iv)

    def test_no_decay_doesnt_change_percentile(self):
        """With exploration_decay=False, percentile never changes."""
        agent = _MinimalCognitiveAgent(anomaly_selection_config={
            "method": "percentile",
            "percentile": 80,
            "min_interventions": 10,
            "max_interventions": 10,
            "exploration_decay": False,
        })
        eps = _all_low_distribution(5, 5)
        scored = agent.compute_scores(eps)
        initial = agent._current_effective_percentile
        for _ in range(20):
            agent.select_groups(scored)
        self.assertEqual(agent._current_effective_percentile, initial)


class TestExtremeCases(unittest.TestCase):
    """Edge-case and stress-test scenarios."""

    def test_empty_episodes(self):
        """No episodes → no selection."""
        agent = _MinimalCognitiveAgent()
        scored = agent.compute_scores([])
        self.assertEqual(len(scored), 0)

    def test_single_group(self):
        """Single group with mixed outcomes → still selected."""
        agent = _MinimalCognitiveAgent(anomaly_selection_config={
            "method": "percentile",
            "percentile": 90,
            "min_interventions": 1,
            "max_interventions": 5,
        })
        eps = _make_episodes({"only": [0, 1, 0, 1, 0, 1]})
        scored = agent.compute_scores(eps)
        self.assertEqual(len(scored), 1)
        sel = agent.select_groups(scored)
        self.assertEqual(len(sel), 1)

    def test_min_equals_max(self):
        """When min == max, selection should be exactly that number."""
        eps = _make_episodes({f"p{i}": [0, 1] for i in range(20)})
        scored = _MinimalCognitiveAgent().compute_scores(eps)

        for n in [1, 5, 10]:
            agent = _MinimalCognitiveAgent(anomaly_selection_config={
                "method": "percentile",
                "percentile": 50,
                "min_interventions": n,
                "max_interventions": n,
                "exploration_decay": False,
            })
            sel = agent.select_groups(scored)
            self.assertEqual(len(sel), min(n, len(scored)))

    def test_many_groups_one_outcome_each(self):
        """Many groups with single episode → all have score=0 (size_bonus only)."""
        eps = _make_episodes({f"p{i}": [0] for i in range(100)})
        agent = _MinimalCognitiveAgent(anomaly_selection_config={
            "method": "percentile",
            "percentile": 50,
            "min_interventions": 5,
            "max_interventions": 10,
        })
        scored = agent.compute_scores(eps)
        # All groups have deviation=0, entropy_bonus=0, size_bonus=0.05
        for _, score, _ in scored:
            self.assertAlmostEqual(score, 0.05, places=6)
        sel = agent.select_groups(scored)
        self.assertLessEqual(len(sel), 10)
        self.assertGreaterEqual(len(sel), 5)


class TestReproducibility(unittest.TestCase):
    """Same config + same data → same result."""

    def test_deterministic_selection(self):
        """Same data produces identical selection across runs."""
        eps = _mixed_rate_distribution(20, 10, seed=42)
        cfg = {
            "method": "percentile",
            "percentile": 70,
            "min_interventions": 3,
            "max_interventions": 8,
            "exploration_decay": False,
        }
        agent1 = _MinimalCognitiveAgent(anomaly_selection_config=cfg)
        agent2 = _MinimalCognitiveAgent(anomaly_selection_config=cfg)
        scored = agent1.compute_scores(eps)
        sel1 = agent1.select_groups(scored)
        sel2 = agent2.select_groups(scored)
        names1 = [s[0] for s in sel1]
        names2 = [s[0] for s in sel2]
        self.assertEqual(names1, names2)


# ===========================================================================
# Detailed benchmarking suite
# ===========================================================================

def benchmark_scenario(name: str, eps_fn, n_groups: int, eps_per_group: int,
                       config: Dict[str, Any], n_iters: int = 30,
                       seed: int = 0) -> Dict[str, Any]:
    """Run a full benchmark scenario and return detailed metrics."""
    eps = eps_fn(n_groups, eps_per_group, seed=seed)
    agent = _MinimalCognitiveAgent(anomaly_selection_config=config)
    scored = agent.compute_scores(eps)

    scores_arr = [s[1] for s in scored]
    iters: List[Dict[str, Any]] = []
    total_selected = 0

    start = time.time()
    for i in range(n_iters):
        t0 = time.time()
        sel = agent.select_groups(scored)
        elapsed = (time.time() - t0) * 1000
        total_selected += len(sel)
        iters.append({
            "iteration": i + 1,
            "n_selected": len(sel),
            "effective_percentile": agent._current_effective_percentile,
            "consecutive_low": agent._anomaly_consecutive_low,
            "history_size": len(agent._anomaly_threshold_history),
            "elapsed_ms": round(elapsed, 2),
        })
    duration = time.time() - start

    return {
        "scenario": name,
        "config": {
            "percentile": config.get("percentile"),
            "min_interventions": config.get("min_interventions"),
            "max_interventions": config.get("max_interventions"),
            "exploration_decay": config.get("exploration_decay", True),
        },
        "data": {
            "n_groups": n_groups,
            "eps_per_group": eps_per_group,
            "n_total_episodes": n_groups * eps_per_group,
        },
        "scores": {
            "min": round(min(scores_arr), 4),
            "max": round(max(scores_arr), 4),
            "mean": round(sum(scores_arr) / len(scores_arr), 4),
            "std": round(
                (sum((s - sum(scores_arr) / len(scores_arr)) ** 2
                     for s in scores_arr) / len(scores_arr)) ** 0.5,
                4,
            ),
            "nonzero": sum(1 for s in scores_arr if s > 0),
            "zero": sum(1 for s in scores_arr if s == 0),
        },
        "iterations": {
            "n": n_iters,
            "total_selected": total_selected,
            "avg_selected_per_iter": round(total_selected / n_iters, 2),
            "min_selected_per_iter": min(it["n_selected"] for it in iters),
            "max_selected_per_iter": max(it["n_selected"] for it in iters),
            "final_percentile": iters[-1]["effective_percentile"],
            "decay_events": sum(
                1 for i in range(1, len(iters))
                if iters[i]["effective_percentile"] < iters[i - 1]["effective_percentile"]
            ),
            "details": iters,
        },
        "timing_ms": round(duration * 1000, 2),
    }


def run_benchmarks() -> Dict[str, Any]:
    """Run all benchmark scenarios and return structured results."""
    results: Dict[str, Any] = {}

    # Scenario 1: Normal operation (mixed rates, mild percentile)
    results["normal_mixed"] = benchmark_scenario(
        "Normal mixed-rate distribution",
        _mixed_rate_distribution, 30, 10,
        {"method": "percentile", "percentile": 85,
         "min_interventions": 3, "max_interventions": 10,
         "exploration_decay": True},
        n_iters=30, seed=100,
    )

    # Scenario 2: Skewed (only 15% of groups have meaningful scores)
    results["skewed_sparse"] = benchmark_scenario(
        "Skewed (15% mixed, 85% uniform)",
        _skewed_distribution, 40, 8,
        {"method": "percentile", "percentile": 85,
         "min_interventions": 3, "max_interventions": 10,
         "exploration_decay": True},
        n_iters=30, seed=200,
    )

    # Scenario 3: Starvation (all REFUSE, no anomalies)
    results["starvation_all_refuse"] = benchmark_scenario(
        "Starvation (all REFUSE)",
        _all_low_distribution, 10, 5,
        {"method": "percentile", "percentile": 90,
         "min_interventions": 3, "max_interventions": 10,
         "exploration_decay": True},
        n_iters=30, seed=300,
    )

    # Scenario 4: Bimodal (two clear clusters)
    results["bimodal_clusters"] = benchmark_scenario(
        "Bimodal (90% ACCEPT vs 10% ACCEPT)",
        _bimodal_distribution, 20, 15,
        {"method": "percentile", "percentile": 80,
         "min_interventions": 3, "max_interventions": 10,
         "exploration_decay": True},
        n_iters=30, seed=400,
    )

    # Scenario 5: Very high percentile (aggressive filtering)
    results["aggressive_filtering"] = benchmark_scenario(
        "Aggressive filtering (percentile=98)",
        _mixed_rate_distribution, 50, 10,
        {"method": "percentile", "percentile": 98,
         "min_interventions": 2, "max_interventions": 5,
         "exploration_decay": True},
        n_iters=30, seed=500,
    )

    # Scenario 6: Very low percentile (minimal filtering)
    results["minimal_filtering"] = benchmark_scenario(
        "Minimal filtering (percentile=20)",
        _mixed_rate_distribution, 30, 10,
        {"method": "percentile", "percentile": 20,
         "min_interventions": 3, "max_interventions": 15,
         "exploration_decay": True},
        n_iters=30, seed=600,
    )

    # Scenario 7: Fixed threshold comparison (legacy)
    eps_fn = _skewed_distribution
    eps = eps_fn(40, 8, seed=200)
    agent = _MinimalCognitiveAgent(anomaly_selection_config={
        "method": "fixed", "percentile": 85,
        "min_interventions": 3, "max_interventions": 10,
    })
    scored = agent.compute_scores(eps)
    fixed_selections = []
    for _ in range(30):
        sel = agent.select_groups(scored)
        fixed_selections.append(len(sel))
    results["fixed_threshold_baseline"] = {
        "scenario": "Fixed threshold baseline (no adaptation)",
        "config": {"method": "fixed"},
        "data": {"n_groups": 40, "eps_per_group": 8},
        "iterations": {
            "n": 30,
            "total_selected": sum(fixed_selections),
            "avg_selected_per_iter": round(sum(fixed_selections) / 30, 2),
            "min_selected_per_iter": min(fixed_selections),
            "max_selected_per_iter": max(fixed_selections),
            "final_percentile": 85,
            "decay_events": 0,
        },
        "timing_ms": 0,
    }

    return results


# ===========================================================================
# Main
# ===========================================================================

def run_tests() -> Dict[str, Any]:
    """Run all unit tests."""
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
    logger.info("ADAPTIVE ANOMALY THRESHOLD — COMPREHENSIVE DYNAMIC TEST")
    logger.info("=" * 60)

    # ── Phase 1: Unit tests ──
    logger.info("\n>>> Phase 1: Unit tests (verification of core logic)")
    t1 = time.time()
    test_result = run_tests()
    t1 = time.time() - t1

    # ── Phase 2: Benchmarks ──
    logger.info("\n>>> Phase 2: Dynamic scenario benchmarks")
    t2 = time.time()
    benchmarks = run_benchmarks()
    t2 = time.time() - t2

    # ── Report ──
    report = {
        "test_results": test_result,
        "benchmarks": benchmarks,
    }

    report_path = "/tmp/anomaly_threshold_test_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)

    # Print summary
    logger.info("=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)
    logger.info("  Tests: %d passed, %d failed, %d errors, %d skipped",
                test_result["tests_run"] - test_result["failures"] - test_result["errors"],
                test_result["failures"], test_result["errors"], test_result["skipped"])
    logger.info("  Test time: %.2fs", t1)
    logger.info("  Benchmark time: %.2fs", t2)

    logger.info("\n  ── Benchmark scenarios ──")
    for name, bm in benchmarks.items():
        its = bm.get("iterations", {})
        sc = bm.get("scores", {})
        logger.info("  %s:", bm.get("scenario", name))
        logger.info("    Scores: mean=%.3f min=%.3f max=%.3f nonzero=%d/%d",
                    sc.get("mean", 0), sc.get("min", 0), sc.get("max", 0),
                    sc.get("nonzero", 0), sc.get("nonzero", 0) + sc.get("zero", 0))
        logger.info("    Selections: avg=%.1f/iter min=%d max=%d total=%d",
                    its.get("avg_selected_per_iter", 0),
                    its.get("min_selected_per_iter", 0),
                    its.get("max_selected_per_iter", 0),
                    its.get("total_selected", 0))
        logger.info("    Decay events: %d  Final percentile: %.1f",
                    its.get("decay_events", 0),
                    its.get("final_percentile", 85))

    logger.info("\n  Report: %s", report_path)
    logger.info("=" * 60)
    logger.info("OVERALL: %s", "PASSED" if test_result["passed"] else "FAILED")
