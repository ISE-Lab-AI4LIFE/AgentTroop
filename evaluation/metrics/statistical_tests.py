"""Statistical inference for HARMONY-X experiments.

Provides:
  - Bootstrap confidence intervals for ASR, MR, program accuracy
  - McNemar's test for paired ASR comparisons (same prompts, two configs)
  - Cohen's kappa for inter-judge / inter-rater agreement
  - Effect size (Cohen's h for proportions)
  - Variance decomposition (within-config vs between-config)

Usage:
    from evaluation.metrics.statistical_tests import (
        bootstrap_ci, mcnemar_test, cohens_kappa,
        cohens_h, variance_decomposition,
    )
"""

from __future__ import annotations

import math
import random
from collections import Counter
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


# ── Bootstrap confidence interval ──────────────────────────────────────────────

def bootstrap_ci(
    values: Sequence[float],
    statistic: str = "mean",
    confidence: float = 0.95,
    n_resamples: int = 10_000,
    random_seed: Optional[int] = None,
) -> Dict[str, float]:
    """Compute bootstrap confidence interval for a given statistic.

    Parameters
    ----------
    values : sequence of float
        Observed metric values (e.g. ASR across seeds).
    statistic : str
        One of ``"mean"``, ``"median"``, ``"std"``.
    confidence : float, default 0.95
        Confidence level (e.g. 0.95 → 95% CI).
    n_resamples : int, default 10_000
        Number of bootstrap resamples.
    random_seed : int, optional
        Seed for reproducibility.

    Returns
    -------
    dict with keys ``"statistic"``, ``"ci_lower"``, ``"ci_upper"``, ``"std_error"``.
    """
    rng = random.Random(random_seed) if random_seed else random.Random()
    arr = np.asarray(values, dtype=float)
    n = len(arr)
    if n == 0:
        return {"statistic": 0.0, "ci_lower": 0.0, "ci_upper": 0.0, "std_error": 0.0}

    stat_fn = {"mean": np.mean, "median": np.median, "std": np.std}[statistic]
    point_est = float(stat_fn(arr))

    boot_stats: List[float] = []
    for _ in range(n_resamples):
        indices = [rng.randint(0, n - 1) for _ in range(n)]
        resample = arr[indices]
        boot_stats.append(float(stat_fn(resample)))

    boot_stats.sort()
    alpha = 1.0 - confidence
    lower_idx = int(round(n_resamples * alpha / 2))
    upper_idx = int(round(n_resamples * (1.0 - alpha / 2)))
    lower = boot_stats[max(0, lower_idx)]
    upper = boot_stats[min(len(boot_stats) - 1, upper_idx)]
    std_err = float(np.std(boot_stats, ddof=1))

    return {
        "statistic": round(point_est, 4),
        "ci_lower": round(lower, 4),
        "ci_upper": round(upper, 4),
        "std_error": round(std_err, 4),
        "confidence_level": confidence,
        "n_resamples": n_resamples,
    }


# ── Paired comparison: McNemar's test ─────────────────────────────────────────

def mcnemar_test(
    outcomes_a: Sequence[int],
    outcomes_b: Sequence[int],
    correction: bool = True,
) -> Dict[str, Any]:
    """McNemar's test for paired nominal data (e.g. ACCEPT/REFUSE on same prompts).

    Tests whether two configurations have different ACCEPT rates on the
    same set of prompts.

    Parameters
    ----------
    outcomes_a : sequence of int
        Binary outcomes for config A (0 = ACCEPT, 1 = REFUSE).
    outcomes_b : sequence of int
        Binary outcomes for config B on the same prompts.
    correction : bool, default True
        Apply Yates's continuity correction.

    Returns
    -------
    dict with keys ``"n"``, ``"n_01"`` (A=0, B=1), ``"n_10"`` (A=1, B=0),
    ``"chi2"``, ``"p_value"``, ``"significant"`` (at α=0.05), ``"correction"``.
    """
    n = len(outcomes_a)
    if n != len(outcomes_b):
        raise ValueError(f"Length mismatch: {len(outcomes_a)} vs {len(outcomes_b)}")

    n_01 = sum(1 for a, b in zip(outcomes_a, outcomes_b) if a == 0 and b == 1)
    n_10 = sum(1 for a, b in zip(outcomes_a, outcomes_b) if a == 1 and b == 0)

    denominator = n_01 + n_10
    if denominator == 0:
        return {
            "n": n,
            "n_01": n_01,
            "n_10": n_10,
            "chi2": 0.0,
            "p_value": 1.0,
            "significant": False,
            "correction": correction,
        }

    if correction:
        chi2 = (abs(n_01 - n_10) - 1) ** 2 / denominator
    else:
        chi2 = (n_01 - n_10) ** 2 / denominator

    from scipy.stats import chi2 as chi2_dist
    p_value = 1.0 - chi2_dist.cdf(chi2, df=1)

    # Effect size: proportion of disagreements where B outperforms A
    effect = (n_01 - n_10) / denominator if denominator > 0 else 0.0

    return {
        "n": n,
        "n_01": n_01,
        "n_10": n_10,
        "chi2": round(chi2, 4),
        "p_value": round(p_value, 6),
        "significant": p_value < 0.05,
        "correction": correction,
        "effect_size": round(effect, 4),
        "interpretation": _mcnemar_interpretation(n_01, n_10, n),
    }


def _mcnemar_interpretation(n_01: int, n_10: int, n: int) -> str:
    """Human-readable interpretation of McNemar result."""
    if n_01 == 0 and n_10 == 0:
        return "No discordant pairs — configurations are identical on this set."
    diff = n_01 - n_10
    if diff > 0:
        pct = abs(diff) / n * 100
        return (
            f"Config B succeeds on {n_01} prompts where A fails; "
            f"A succeeds on {n_10} where B fails. "
            f"Net advantage to B: {diff}/{n} ({pct:.1f}%)."
        )
    elif diff < 0:
        pct = abs(diff) / n * 100
        return (
            f"Config A succeeds on {n_10} prompts where B fails; "
            f"B succeeds on {n_01} where A fails. "
            f"Net advantage to A: {-diff}/{n} ({pct:.1f}%)."
        )
    return "Discordant pairs are balanced — no net difference."


# ── Inter-rater agreement: Cohen's kappa ──────────────────────────────────────

def cohens_kappa(
    ratings_a: Sequence[int],
    ratings_b: Sequence[int],
    weights: Optional[str] = None,
) -> Dict[str, Any]:
    """Cohen's kappa for inter-rater agreement between two judges.

    Parameters
    ----------
    ratings_a, ratings_b : sequence of int
        Ratings from two judges on the same items.
    weights : str, optional
        ``"linear"`` or ``"quadratic"`` for ordinal weights.

    Returns
    -------
    dict with keys ``"kappa"``, ``"p_o"`` (observed agreement),
    ``"p_e"`` (expected agreement), ``"n"``, ``"agreement_pct"``.
    """
    n = len(ratings_a)
    if n != len(ratings_b):
        raise ValueError(f"Length mismatch: {len(ratings_a)} vs {len(ratings_b)}")
    if n == 0:
        return {"kappa": 0.0, "p_o": 0.0, "p_e": 0.0, "n": 0, "agreement_pct": 0.0}

    from sklearn.metrics import cohen_kappa_score
    kappa = float(cohen_kappa_score(ratings_a, ratings_b, weights=weights))

    # Observed agreement
    n_agree = sum(1 for a, b in zip(ratings_a, ratings_b) if a == b)
    p_o = n_agree / n

    # Expected agreement (marginal-independent)
    count_a = Counter(ratings_a)
    count_b = Counter(ratings_b)
    all_labels = set(count_a.keys()) | set(count_b.keys())
    p_e = sum(
        (count_a.get(label, 0) / n) * (count_b.get(label, 0) / n)
        for label in all_labels
    )
    p_e = max(p_e, 1e-10)  # prevent division by zero

    return {
        "kappa": round(kappa, 4),
        "p_o": round(p_o, 4),
        "p_e": round(p_e, 4),
        "n": n,
        "agreement_pct": round(p_o * 100, 2),
        "strength": _kappa_strength(kappa),
    }


def _kappa_strength(kappa: float) -> str:
    if kappa >= 0.81:
        return "Almost perfect"
    elif kappa >= 0.61:
        return "Substantial"
    elif kappa >= 0.41:
        return "Moderate"
    elif kappa >= 0.21:
        return "Fair"
    elif kappa >= 0.0:
        return "Slight"
    return "Poor (worse than chance)"


# ── Effect size for proportions: Cohen's h ────────────────────────────────────

def cohens_h(prop_a: float, prop_b: float) -> Dict[str, float]:
    """Cohen's h for the difference between two proportions.

    h = 2 * arcsin(sqrt(p1)) - 2 * arcsin(sqrt(p2))

    Interpretation (Cohen, 1988):
        h = 0.2 → small
        h = 0.5 → medium
        h = 0.8 → large
    """
    p1 = max(0.0, min(1.0, prop_a))
    p2 = max(0.0, min(1.0, prop_b))
    h = 2.0 * math.asin(math.sqrt(p1)) - 2.0 * math.asin(math.sqrt(p2))
    magnitude = abs(h)
    if magnitude >= 0.8:
        desc = "large"
    elif magnitude >= 0.5:
        desc = "medium"
    elif magnitude >= 0.2:
        desc = "small"
    else:
        desc = "negligible"
    return {
        "h": round(h, 4),
        "prop_a": round(p1, 4),
        "prop_b": round(p2, 4),
        "magnitude": round(magnitude, 4),
        "interpretation": desc,
    }


# ── Variance decomposition ─────────────────────────────────────────────────────

def variance_decomposition(
    results: Dict[str, Sequence[float]],
) -> Dict[str, Any]:
    """Decompose total variance into within-config and between-config components.

    Parameters
    ----------
    results : dict of str → list of float
        Keys are config labels, values are metric values across seeds.

    Returns
    -------
    dict with keys ``"total_variance"``, ``"within_config_variance"``,
    ``"between_config_variance"``, ``"icc"`` (intraclass correlation),
    ``"n_configs"``, ``"n_total"``.
    """
    configs = list(results.keys())
    groups = [np.asarray(results[c], dtype=float) for c in configs]
    n_configs = len(configs)
    n_total = sum(len(g) for g in groups)

    if n_total == 0 or n_configs < 2:
        return {
            "total_variance": 0.0,
            "within_config_variance": 0.0,
            "between_config_variance": 0.0,
            "icc": 0.0,
            "n_configs": n_configs,
            "n_total": n_total,
        }

    all_values = np.concatenate(groups)
    grand_mean = np.mean(all_values)

    # Between-config variance (MSB)
    ss_between = sum(
        len(g) * (np.mean(g) - grand_mean) ** 2 for g in groups
    )
    ms_between = ss_between / (n_configs - 1) if n_configs > 1 else 0.0

    # Within-config variance (MSW)
    ss_within = sum(np.sum((g - np.mean(g)) ** 2) for g in groups)
    ms_within = ss_within / (n_total - n_configs) if n_total > n_configs else 0.0

    total_var = np.var(all_values, ddof=0)

    # ICC(1) — proportion of total variance due to config differences
    # Using the formula: ICC = (MSB - MSW) / (MSB + (k_bar - 1) * MSW)
    # where k_bar is average group size
    k_bar = np.mean([len(g) for g in groups])
    icc = (ms_between - ms_within) / (
        ms_between + (k_bar - 1) * ms_within
    ) if (ms_between + (k_bar - 1) * ms_within) > 0 else 0.0

    return {
        "total_variance": round(float(total_var), 6),
        "within_config_variance": round(float(ms_within), 6),
        "between_config_variance": round(float(ms_between), 6),
        "icc": round(float(icc), 4),
        "n_configs": n_configs,
        "n_total": n_total,
        "interpretation": _icc_interpretation(icc),
    }


def _icc_interpretation(icc: float) -> str:
    if icc >= 0.75:
        return "Config differences dominate — ablations matter."
    elif icc >= 0.50:
        return "Config differences substantial but with notable within-config noise."
    elif icc >= 0.25:
        return "Moderate config effect; high within-seed variability."
    else:
        return "Within-seed noise comparable to or larger than config effect."


# ── Summary builder ────────────────────────────────────────────────────────────

def build_ablation_summary(
    seed_results: Dict[str, List[Dict[str, Any]]],
    metrics: Sequence[str] = ("asr", "program_accuracy", "convergence_speed"),
) -> Dict[str, Any]:
    """Build a complete statistical summary across ablation configs.

    Parameters
    ----------
    seed_results : dict of str → list of dict
        Keys are config names, values are per-seed result dicts.
        Each result dict must contain the requested metric keys.
    metrics : sequence of str
        Metric names to summarise (e.g. ``"asr"``, ``"program_accuracy"``).

    Returns
    -------
    dict with keys ``"configs"``, ``"metrics"``, ``"summary"``, ``"variance_decomp"``.
    """
    config_names = list(seed_results.keys())
    output: Dict[str, Any] = {
        "configs": config_names,
        "metrics": list(metrics),
        "summary": {},
        "variance_decomp": {},
    }

    # Per-config summary
    for cfg in config_names:
        output["summary"][cfg] = {}
        for metric in metrics:
            values = [r.get(metric, 0.0) for r in seed_results[cfg]]
            ci = bootstrap_ci(values, statistic="mean", random_seed=42)
            output["summary"][cfg][metric] = ci

    # Variance decomposition per metric
    for metric in metrics:
        grouped = {
            cfg: [r.get(metric, 0.0) for r in seed_results[cfg]]
            for cfg in config_names
        }
        output["variance_decomp"][metric] = variance_decomposition(grouped)

    # Pairwise McNemar for ASR (if per-prompt outcomes available)
    output["pairwise_tests"] = {}
    for i, cfg_a in enumerate(config_names):
        for cfg_b in config_names[i + 1:]:
            key = f"{cfg_a}_vs_{cfg_b}"
            pair_key = f"asr"
            # Check if per-prompt outcomes stored
            outcomes_a = [
                r.get("per_prompt_outcomes", [])
                for r in seed_results[cfg_a]
            ]
            outcomes_b = [
                r.get("per_prompt_outcomes", [])
                for r in seed_results[cfg_b]
            ]
            # Use first seed's outcomes if available as example
            if outcomes_a and outcomes_b and outcomes_a[0] and outcomes_b[0]:
                if len(outcomes_a[0]) == len(outcomes_b[0]):
                    mcnemar = mcnemar_test(outcomes_a[0], outcomes_b[0])
                    output["pairwise_tests"][key] = mcnemar

    return output
