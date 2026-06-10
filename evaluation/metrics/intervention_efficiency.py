from __future__ import annotations

import logging
from typing import Callable, Optional

from knowledge.episodic import EpisodicMemory, EpisodeFilter

logger = logging.getLogger(__name__)


class InterventionEfficiencyMetric:
    """RQ1: Number of interventions needed to reach a target accuracy threshold.

    Reads episodes from EpisodicMemory in chronological order and determines
    the point at which the program accuracy first exceeds the given threshold.
    When ``predict_fn`` is provided, the metric measures the program's running
    prediction accuracy against ground-truth episode outcomes; otherwise it
    falls back to the victim's REFUSE rate.

    Supports optional baseline comparison for random probing or no-targeted-intervention runs.
    """

    def __init__(self, episodic_memory: EpisodicMemory) -> None:
        self._memory = episodic_memory

    def _compute_campaign_metrics(
        self,
        campaign_id: str,
        experiment_id: Optional[str],
        threshold: float,
        predict_fn: Optional[Callable[[str], int]] = None,
    ) -> dict:
        filter_kwargs = {"campaign_id": campaign_id}
        if experiment_id is not None:
            filter_kwargs["experiment_id"] = experiment_id
        ep_filter = EpisodeFilter(**filter_kwargs)
        episodes = self._memory.filter_episodes(ep_filter)
        episodes.sort(key=lambda e: e.created_at)

        if not episodes:
            logger.warning("No episodes for campaign=%s", campaign_id)
            return {
                "campaign_id": campaign_id,
                "total_episodes": 0,
                "interventions_to_threshold": -1,
                "best_accuracy": 0.0,
                "threshold": threshold,
                "reached": False,
            }

        total = len(episodes)
        reached = False
        intv_to_thresh = total
        best_accuracy = 0.0

        if predict_fn is not None:
            # ── Program prediction accuracy mode ──
            # Measure how quickly the learned program correctly predicts
            # episode outcomes.
            for i in range(1, total + 1):
                window = episodes[:i]
                correct = 0
                total_in_window = 0
                for ep in window:
                    if ep.outcome is None:
                        continue
                    try:
                        pred = predict_fn(ep.intervention.prompt)
                    except Exception:
                        pred = 0
                    if pred == ep.outcome:
                        correct += 1
                    total_in_window += 1
                if total_in_window == 0:
                    continue
                accuracy = correct / total_in_window
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                if accuracy >= threshold:
                    intv_to_thresh = i
                    reached = True
                    break
        else:
            # ── Victim REFUSE rate mode (legacy) ──
            all_outcomes = [ep.outcome for ep in episodes if ep.outcome is not None]
            n_refuse = sum(1 for o in all_outcomes if o == 1)
            n_accept = len(all_outcomes) - n_refuse

            if n_refuse == 0 or n_accept == 0:
                logger.info(
                    "Insufficient outcome diversity: refuse=%d accept=%d (need both)",
                    n_refuse, n_accept,
                )
                return {
                    "campaign_id": campaign_id,
                    "total_episodes": total,
                    "interventions_to_threshold": -1,
                    "best_accuracy": 0.0,
                    "threshold": threshold,
                    "reached": False,
                    "n_refuse": n_refuse,
                    "n_accept": n_accept,
                    "note": "Insufficient outcome diversity — need both ACCEPT and REFUSE",
                }

            for i in range(1, total + 1):
                window = episodes[:i]
                outcomes = [ep.outcome for ep in window if ep.outcome is not None]
                if not outcomes:
                    continue
                positives = sum(1 for o in outcomes if o == 1)
                accuracy = positives / len(outcomes)
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                if accuracy >= threshold:
                    intv_to_thresh = i
                    reached = True
                    break

        logger.info(
            "InterventionEfficiency: campaign=%s threshold=%.2f reached=%s at=%d eps best=%.4f total=%d",
            campaign_id, threshold, reached, intv_to_thresh, best_accuracy, total,
        )
        return {
            "campaign_id": campaign_id,
            "total_episodes": total,
            "interventions_to_threshold": intv_to_thresh,
            "best_accuracy": best_accuracy,
            "threshold": threshold,
            "reached": reached,
        }

    def compute(
        self,
        campaign_id: str,
        experiment_id: Optional[str] = None,
        threshold: float = 0.85,
        predict_fn: Optional[Callable[[str], int]] = None,
        baseline_campaign_id: Optional[str] = None,
        baseline_experiment_id: Optional[str] = None,
    ) -> dict:
        result = self._compute_campaign_metrics(campaign_id, experiment_id, threshold, predict_fn)

        if baseline_campaign_id:
            baseline = self._compute_campaign_metrics(
                baseline_campaign_id,
                baseline_experiment_id,
                threshold,
            )
            result.update({
                "baseline_campaign_id": baseline_campaign_id,
                "baseline_experiment_id": baseline_experiment_id,
                "baseline_total_episodes": baseline["total_episodes"],
                "baseline_interventions_to_threshold": baseline["interventions_to_threshold"],
                "baseline_best_accuracy": baseline["best_accuracy"],
                "baseline_reached": baseline["reached"],
            })
            if baseline["interventions_to_threshold"] > 0:
                result["improvement_ratio"] = (
                    1.0
                    - result["interventions_to_threshold"]
                    / baseline["interventions_to_threshold"]
                )
            else:
                result["improvement_ratio"] = 0.0

        return result
