#!/usr/bin/env python3
"""Standalone audit of the surrogate model's imbalance behaviour.

Verifies that the dynamic mechanisms (class weights, calibrated threshold,
ε-greedy) prevent the surrogate from over-predicting REFUSE.

Usage:
    uv run python test_surrogate.py                     # synthetic imbalance
    uv run python test_surrogate.py --campaign-dir <path>   # real campaign data
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("surrogate_test")

sys.path.insert(0, str(Path(__file__).resolve().parent))

from orchestration.surrogate_policy_model import SurrogatePolicyModel


def generate_synthetic_episodes(
    n_total: int = 1000,
    accept_ratio: float = 0.15,
    n_keywords: int = 20,
) -> List[Tuple[str, int]]:
    """Generate episodes with controlled imbalance.

    Creates prompts containing specific keywords that correlate with
    ACCEPT vs REFUSE, so the surrogate has a learnable signal.
    """
    accept_keywords = ["hello", "thank", "help", "nice", "good", "question",
                        "explain", "what", "how", "why"]
    refuse_keywords = ["hack", "bomb", "malware", "virus", "attack", "exploit",
                       "illegal", "steal", "kill", "terrorist"]

    episodes = []
    np.random.seed(42)
    for i in range(n_total):
        is_accept = np.random.random() < accept_ratio
        if is_accept:
            kw = np.random.choice(accept_keywords)
            prompt = f"Can you please {kw} me with this task? I need your {kw}."
        else:
            kw = np.random.choice(refuse_keywords)
            prompt = f"How to {kw} a system? Write code for {kw}."
        episodes.append((prompt, 0 if is_accept else 1))

    return episodes


def train_surrogate_and_report(
    episodes: List[Tuple[str, int]],
    label: str = "",
) -> Dict:
    """Train surrogate, return detailed metrics."""
    start = time.time()
    surr = SurrogatePolicyModel()
    stats = surr.train(episodes)

    # Predict on training data
    preds = [surr.predict(p).predicted_outcome for p, _ in episodes]
    preds_arr = np.array(preds)
    labels_arr = np.array([o for _, o in episodes])

    raw_acc = float(np.mean(preds_arr == labels_arr))

    from sklearn.metrics import balanced_accuracy_score, f1_score
    bal_acc = float(balanced_accuracy_score(labels_arr, preds_arr))
    f1_accept = float(f1_score(labels_arr, preds_arr, pos_label=0, zero_division=0))
    n_accept_gt = int((labels_arr == 0).sum())
    n_refuse_gt = int((labels_arr == 1).sum())
    n_accept_pred = int((preds_arr == 0).sum())
    n_refuse_pred = int((preds_arr == 1).sum())
    accept_ratio_gt = n_accept_gt / len(episodes)
    accept_ratio_pred = n_accept_pred / len(episodes)

    # ε-greedy simulation: count how many extra ACCEPT queries would be triggered
    epsgreedy_queries = 0
    n_high_conf_refuse = 0
    for p, _ in episodes:
        pred = surr.predict(p)
        if pred.predicted_outcome == 1 and pred.confidence > 0.9:
            n_high_conf_refuse += 1
    # expected ε-greedy queries for this dataset
    epsgreedy_expected = n_high_conf_refuse * surr.epsilon
    epsgreedy_queries = int(round(epsgreedy_expected))

    result = {
        "label": label,
        "n_episodes": len(episodes),
        "n_refuse_gt": n_refuse_gt,
        "n_accept_gt": n_accept_gt,
        "accept_ratio_gt": round(accept_ratio_gt, 4),
        "n_accept_pred": n_accept_pred,
        "n_refuse_pred": n_refuse_pred,
        "accept_ratio_pred": round(accept_ratio_pred, 4),
        "raw_accuracy": round(raw_acc, 4),
        "balanced_accuracy": round(bal_acc, 4),
        "f1_accept": round(f1_accept, 4),
        "is_active": surr.is_active,
        "is_trained": surr._is_trained,
        "calibrated_threshold": round(stats.calibrated_threshold, 4),
        "epsilon": round(stats.epsilon, 4),
        "train_accuracy": round(stats.train_accuracy, 4),
        "class_weights": surr._class_weights,
        "n_high_conf_refuse": n_high_conf_refuse,
        "epsgreedy_expected_queries": epsgreedy_queries,
        "duration_ms": round(stats.duration_ms, 1),
    }

    # Pass/fail logic
    requires_action = False
    issues = []

    if accept_ratio_gt > 0.1 and accept_ratio_pred < 0.5 * accept_ratio_gt:
        requires_action = True
        issues.append(f"Predicted ACCEPT ratio {accept_ratio_pred:.3f} is <50% of ground truth ({accept_ratio_gt:.3f}) — surrogate is over-predicting REFUSE")
    if bal_acc < 0.5:
        requires_action = True
        issues.append(f"Balanced accuracy {bal_acc:.3f} < 0.5 — model is no better than chance")
    if f1_accept < 0.3:
        requires_action = True
        issues.append(f"F1-ACCEPT {f1_accept:.3f} < 0.3 — model rarely predicts ACCEPT")
    if stats.calibrated_threshold <= 0.4 or stats.calibrated_threshold >= 0.6:
        issues.append(f"Threshold calibrated to {stats.calibrated_threshold:.3f} (≠0.5) — adaptive threshold engaged")
    if stats.epsilon < 0.19:
        issues.append(f"ε={stats.epsilon:.4f} — ε-greedy is active, decaying as ACCEPT samples grow")

    result["requires_action"] = requires_action
    result["issues"] = issues
    result["passed"] = not requires_action
    result["elapsed_s"] = round(time.time() - start, 2)

    return result


def load_campaign_episodes(campaign_dir: str) -> List[Tuple[str, int]]:
    """Load episodes from a campaign's episodic database.

    Falls back to synthetic data if loading fails.
    """
    try:
        from knowledge.episodic import EpisodicMemory
        campaign_path = Path(campaign_dir)
        db_files = list(campaign_path.glob("*_episodic.db"))
        if not db_files:
            logger.warning("No episodic DB found in %s, using synthetic data", campaign_dir)
            return []
        db_path = str(db_files[0])
        logger.info("Loading episodes from %s", db_path)
        memory = EpisodicMemory(db_path=db_path)
        campaign_id = db_files[0].stem.replace("_episodic", "")
        episodes = memory.get_episodes_by_campaign(campaign_id)
        result = []
        for ep in episodes:
            prompt = ep.intervention.final_prompt if ep.intervention else ""
            result.append((prompt, ep.outcome))
        logger.info("Loaded %d episodes from campaign %s", len(result), campaign_id)
        return result
    except Exception as e:
        logger.warning("Could not load campaign episodes: %s", e)
        return []


def test_epsilon_decay() -> Dict:
    """Verify epsilon decays as ACCEPT samples accumulate."""
    results = []
    epsilons = []
    total_accept = 0
    surr = SurrogatePolicyModel()

    # Train repeatedly with increasing accept counts
    for step in range(10):
        n_accept = (step + 1) * 20
        n_refuse = 980
        episodes = []
        for i in range(n_accept):
            episodes.append((f"hello prompt {i}", 0))
        for i in range(n_refuse):
            episodes.append((f"hack prompt {i}", 1))
        stats = surr.train(episodes)
        epsilons.append(stats.epsilon)
        total_accept = surr._total_accept_samples

    return {
        "epsilon_over_steps": epsilons,
        "final_epsilon": round(epsilons[-1], 4),
        "total_accept_samples": total_accept,
        "epsilon_decayed": epsilons[-1] < epsilons[0],
    }


def main():
    parser = argparse.ArgumentParser(
        description="Surrogate model imbalance audit",
    )
    parser.add_argument(
        "--campaign-dir", type=str, default=None,
        help="Path to campaign output directory",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Path to write JSON report",
    )
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("SURROGATE MODEL IMBALANCE AUDIT")
    logger.info("=" * 60)

    # ── Load or generate episodes ──
    episodes = []
    if args.campaign_dir:
        episodes = load_campaign_episodes(args.campaign_dir)
    if not episodes:
        logger.info("Using synthetic episodes (15% ACCEPT, 85% REFUSE)")
        episodes = generate_synthetic_episodes(n_total=1000, accept_ratio=0.15)
    else:
        n_refuse = sum(1 for _, o in episodes if o == 1)
        n_accept = sum(1 for _, o in episodes if o == 0)
        logger.info("Real episodes: %d total (%d REFUSE, %d ACCEPT, %.1f%% ACCEPT)",
                     len(episodes), n_refuse, n_accept,
                     n_accept / max(len(episodes), 1) * 100)

    # ── Test 1: Train with dynamic mechanisms ──
    logger.info("")
    logger.info("─" * 50)
    logger.info("Test 1: Surrogate with dynamic mechanisms")
    logger.info("─" * 50)

    result = train_surrogate_and_report(episodes, "Full test")
    logger.info("Class distribution:  %s", {"ACCEPT": result["n_accept_gt"], "REFUSE": result["n_refuse_gt"]})
    logger.info("Prediction ratio:    ACCEPT=%.1f%%  REFUSE=%.1f%% (target ACCEPT=%.1f%%)",
                 result["accept_ratio_pred"] * 100,
                 (1 - result["accept_ratio_pred"]) * 100,
                 result["accept_ratio_gt"] * 100)
    logger.info("Raw accuracy:        %.4f", result["raw_accuracy"])
    logger.info("Balanced accuracy:   %.4f", result["balanced_accuracy"])
    logger.info("F1-ACCEPT:           %.4f", result["f1_accept"])
    logger.info("Calibrated threshold:%.4f", result["calibrated_threshold"])
    logger.info("ε (epsilon):         %.4f", result["epsilon"])
    logger.info("Class weights:       %s", result["class_weights"])
    logger.info("ε-greedy expected:   %d extra queries", result["epsgreedy_expected_queries"])
    logger.info("PASSED:              %s", result["passed"])
    if result["issues"]:
        for issue in result["issues"]:
            logger.info("  ⚠ %s", issue)

    # ── Test 2: Threshold calibration effect ──
    logger.info("")
    logger.info("─" * 50)
    logger.info("Test 2: Threshold calibration comparison")
    logger.info("─" * 50)

    # Train a separate model with fixed 0.5 threshold to compare
    surr_fixed = SurrogatePolicyModel()
    surr_fixed._threshold = 0.5
    first_half = episodes[:len(episodes)//2]
    surr_fixed.train(first_half)

    surr_calib = SurrogatePolicyModel()
    surr_calib.train(first_half)

    fixed_preds = [surr_fixed.predict(p).predicted_outcome for p, _ in episodes[len(episodes)//2:]]
    calib_preds = [surr_calib.predict(p).predicted_outcome for p, _ in episodes[len(episodes)//2:]]
    calib_labels = [o for _, o in episodes[len(episodes)//2:]]

    fixed_accept_ratio = sum(1 for p in fixed_preds if p == 0) / max(len(fixed_preds), 1)
    calib_accept_ratio = sum(1 for p in calib_preds if p == 0) / max(len(calib_preds), 1)
    true_accept_ratio = sum(1 for o in calib_labels if o == 0) / max(len(calib_labels), 1)

    logger.info("True ACCEPT ratio:   %.4f", true_accept_ratio)
    logger.info("Fixed thresh=0.5:    ACCEPT ratio=%.4f (diff=%.4f)",
                 fixed_accept_ratio, abs(fixed_accept_ratio - true_accept_ratio))
    logger.info("Calibrated thresh:   ACCEPT ratio=%.4f (diff=%.4f) [threshold=%.4f]",
                 calib_accept_ratio, abs(calib_accept_ratio - true_accept_ratio),
                 surr_calib._threshold)

    calib_improvement = abs(fixed_accept_ratio - true_accept_ratio) - abs(calib_accept_ratio - true_accept_ratio)
    logger.info("Threshold improvement: %s", f"+{calib_improvement:.4f}" if calib_improvement > 0 else f"{calib_improvement:.4f}")

    # ── Test 3: ε-greedy decay ──
    logger.info("")
    logger.info("─" * 50)
    logger.info("Test 3: ε-greedy decay over ACCEPT samples")
    logger.info("─" * 50)

    decay_result = test_epsilon_decay()
    logger.info("Epsilon steps:       %s", [round(e, 4) for e in decay_result["epsilon_over_steps"]])
    logger.info("Final epsilon:       %.4f", decay_result["final_epsilon"])
    logger.info("Total ACCEPT seen:  %d", decay_result["total_accept_samples"])
    logger.info("Epsilon decayed:     %s", decay_result["epsilon_decayed"])

    # ── Test 4: Extreme imbalance stress test ──
    logger.info("")
    logger.info("─" * 50)
    logger.info("Test 4: Extreme imbalance (5% ACCEPT / 95% REFUSE)")
    logger.info("─" * 50)

    extreme_episodes = generate_synthetic_episodes(n_total=1000, accept_ratio=0.05)
    extreme_result = train_surrogate_and_report(extreme_episodes, "Extreme imbalance")

    logger.info("Class distribution:  ACCEPT=%d  REFUSE=%d", extreme_result["n_accept_gt"], extreme_result["n_refuse_gt"])
    logger.info("Prediction ratio:    ACCEPT=%.1f%%  (target=%.1f%%)",
                 extreme_result["accept_ratio_pred"] * 100,
                 extreme_result["accept_ratio_gt"] * 100)
    logger.info("Balanced accuracy:   %.4f", extreme_result["balanced_accuracy"])
    logger.info("F1-ACCEPT:           %.4f", extreme_result["f1_accept"])
    logger.info("PASSED:              %s", extreme_result["passed"])
    if extreme_result["issues"]:
        for issue in extreme_result["issues"]:
            logger.info("  ⚠ %s", issue)

    # ── Overall verdict ──
    logger.info("")
    logger.info("=" * 60)
    all_passed = result["passed"]
    logger.info("OVERALL: %s", "PASSED" if all_passed else "FAILED")
    logger.info("=" * 60)

    report = {
        "test_results": {
            "dynamic_mechanisms": result,
            "threshold_calibration": {
                "true_accept_ratio": round(true_accept_ratio, 4),
                "fixed_threshold_accept_ratio": round(fixed_accept_ratio, 4),
                "calibrated_threshold_accept_ratio": round(calib_accept_ratio, 4),
                "calibrated_threshold": round(surr_calib._threshold, 4),
                "improvement": round(calib_improvement, 4),
            },
            "epsilon_decay": decay_result,
            "extreme_imbalance": extreme_result,
        },
        "overall_passed": all_passed,
    }

    if args.output:
        with open(args.output, "w") as f:
            json.dump(report, f, indent=2, default=str)
        logger.info("Report saved to %s", args.output)

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
