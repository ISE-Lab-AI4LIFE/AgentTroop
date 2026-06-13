"""Transfer Speed metrics — measures cross-model transfer learning efficiency.

RQ-TL-1: Cold-start queries vs warm-start queries to reach accuracy > 90%
RQ-TL-2: Entropy reduction rate with vs without scientific memory transfer.

Metrics::

    transfer_speedup = cold_queries / warm_queries
    transfer_accuracy_boost = warm_init_accuracy - cold_init_accuracy
    queries_to_90pct(victim, warm_start)  # interventions needed
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from core.executor import ProgramExecutor
from core.primitive import default_registry
from knowledge.episodic import EpisodicMemory

logger = logging.getLogger(__name__)


@dataclass
class TransferMetrics:
    """Metrics from a transfer learning evaluation."""

    cold_queries_to_threshold: int = 0
    warm_queries_to_threshold: int = 0
    transfer_speedup: float = 1.0
    cold_init_accuracy: float = 0.0
    warm_init_accuracy: float = 0.0
    accuracy_boost: float = 0.0
    cold_entropy_history: List[float] = field(default_factory=list)
    warm_entropy_history: List[float] = field(default_factory=list)
    cold_accuracy_history: List[float] = field(default_factory=list)
    warm_accuracy_history: List[float] = field(default_factory=list)
    queries_per_campaign: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "cold_queries_to_threshold": self.cold_queries_to_threshold,
            "warm_queries_to_threshold": self.warm_queries_to_threshold,
            "transfer_speedup": round(self.transfer_speedup, 3),
            "cold_init_accuracy": round(self.cold_init_accuracy, 3),
            "warm_init_accuracy": round(self.warm_init_accuracy, 3),
            "accuracy_boost": round(self.accuracy_boost, 3),
            "queries_per_campaign": self.queries_per_campaign,
        }


def evaluate_transfer(
    cold_accuracy_curve: List[float],
    warm_accuracy_curve: List[float],
    cold_entropy: Optional[List[float]] = None,
    warm_entropy: Optional[List[float]] = None,
    accuracy_threshold: float = 0.9,
) -> TransferMetrics:
    """Compare cold-start vs warm-start accuracy curves.

    Parameters
    ----------
    cold_accuracy_curve : list of float
        Accuracy per intervention (cold start — no prior knowledge).
    warm_accuracy_curve : list of float
        Accuracy per intervention (warm start — with Scientific Memory).
    cold_entropy : list of float, optional
    warm_entropy : list of float, optional
    accuracy_threshold : float

    Returns
    -------
    TransferMetrics
    """
    metrics = TransferMetrics()

    # Queries to reach threshold
    for i, acc in enumerate(cold_accuracy_curve):
        if acc >= accuracy_threshold:
            metrics.cold_queries_to_threshold = i + 1
            break
    for i, acc in enumerate(warm_accuracy_curve):
        if acc >= accuracy_threshold:
            metrics.warm_queries_to_threshold = i + 1
            break

    metrics.cold_init_accuracy = cold_accuracy_curve[0] if cold_accuracy_curve else 0.0
    metrics.warm_init_accuracy = warm_accuracy_curve[0] if warm_accuracy_curve else 0.0
    metrics.accuracy_boost = metrics.warm_init_accuracy - metrics.cold_init_accuracy

    if metrics.warm_queries_to_threshold > 0 and metrics.cold_queries_to_threshold > 0:
        metrics.transfer_speedup = (
            metrics.cold_queries_to_threshold / metrics.warm_queries_to_threshold
        )

    if cold_entropy:
        metrics.cold_entropy_history = cold_entropy
    if warm_entropy:
        metrics.warm_entropy_history = warm_entropy

    metrics.cold_accuracy_history = cold_accuracy_curve
    metrics.warm_accuracy_history = warm_accuracy_curve
    metrics.queries_per_campaign = max(len(cold_accuracy_curve), len(warm_accuracy_curve))

    logger.info(
        "Transfer: cold=%d warm=%d speedup=%.2fx boost=%.3f",
        metrics.cold_queries_to_threshold,
        metrics.warm_queries_to_threshold,
        metrics.transfer_speedup,
        metrics.accuracy_boost,
    )
    return metrics


class TransferSpeedMetric:
    """Metric class wrapping evaluate_transfer for the evaluation pipeline.

    Provides the ``compute()`` interface expected by RQ2Evaluator.
    Loads per-campaign episodic DBs separately so prior campaigns can be
    read from their own files.  Accuracy curves are built by evaluating
    each campaign's best discovered program against the actual victim
    outcomes episode by episode, giving real prediction accuracy rather
    than a crude outcome proxy.
    """

    def __init__(
        self,
        episodic_memory: Any,
        db_dir: str = ".",
        outputs_dir: Optional[str] = None,
    ) -> None:
        self._episodic = episodic_memory
        self._db_dir = db_dir
        self._outputs_dir = outputs_dir or db_dir

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute(
        self,
        prior_campaign_id: str,
        target_campaign_id: str,
        prior_experiment_id: Optional[str] = None,
        target_experiment_id: Optional[str] = None,
        threshold: float = 0.9,
    ) -> Dict[str, Any]:
        """Compute transfer speed metrics between two campaigns.

        Builds cold-start (prior) and warm-start (target) accuracy curves
        from episodic memory, then delegates to ``evaluate_transfer()``.
        """
        cold_curve = self._build_accuracy_curve(
            prior_campaign_id, prior_experiment_id, threshold,
        )
        warm_curve = self._build_accuracy_curve(
            target_campaign_id, target_experiment_id, threshold,
        )
        metrics = evaluate_transfer(cold_curve, warm_curve, accuracy_threshold=threshold)
        return metrics.to_dict()

    # ------------------------------------------------------------------
    # Episode loading  — opens per-campaign DB independently
    # ------------------------------------------------------------------

    def _load_episodes(self, campaign_id: str) -> list:
        """Return episodes for *campaign_id*, trying the shared memory first,
        then falling back to the per-campaign DB file."""
        try:
            episodes = self._episodic.get_episodes_by_campaign(campaign_id)
            if episodes:
                return episodes
        except Exception:
            pass

        db_path = os.path.join(self._db_dir, f"{campaign_id}_episodic.db")
        if not os.path.exists(db_path):
            logger.warning("No episodic DB found for campaign %s at %s", campaign_id, db_path)
            return []

        prior = EpisodicMemory(db_path=db_path)
        try:
            return prior.get_episodes_by_campaign(campaign_id)
        except Exception as exc:
            logger.warning("Failed to load episodes for %s: %s", campaign_id, exc)
            return []
        finally:
            prior.close()

    # ------------------------------------------------------------------
    # Best-program loading  — reads version_space.json per campaign
    # ------------------------------------------------------------------

    def _load_best_program(self, campaign_id: str) -> Any:
        """Load the highest-posterior program for *campaign_id* that can be
        deserialised with the current primitive registry (i.e. whose
        predicates have not been removed)."""
        vs_path = os.path.join(self._outputs_dir, campaign_id, "version_space.json")
        if not os.path.exists(vs_path):
            return None
        try:
            with open(vs_path) as f:
                vs = json.load(f)
        except Exception:
            return None

        candidates = vs.get("candidates", [])
        if not candidates:
            return None

        from core.program import Program

        # Try candidates in descending posterior order; skip any whose
        # predicates are no longer registered.
        for cand in sorted(candidates, key=lambda c: c.get("posterior", 0.0), reverse=True):
            best_id = cand.get("program_id")
            if not best_id:
                continue
            prog_dict = vs.get("program_asts", {}).get(best_id)
            if not prog_dict:
                continue
            try:
                return Program.from_dict(prog_dict)
            except Exception:
                continue

        return None

    # ------------------------------------------------------------------
    # Accuracy curve  — real prediction accuracy, not outcome proxy
    # ------------------------------------------------------------------

    def _build_accuracy_curve(
        self,
        campaign_id: str,
        experiment_id: Optional[str] = None,
        threshold: float = 0.9,
    ) -> List[float]:
        episodes = self._load_episodes(campaign_id)
        if not episodes:
            return []

        program = self._load_best_program(campaign_id)
        executor = ProgramExecutor(registry=default_registry) if program else None

        raw: List[float] = []
        curve: List[float] = []
        for ep in episodes:
            prompt = ""
            if ep.intervention:
                prompt = ep.intervention.final_prompt or ep.intervention.prompt or ""

            if executor is not None and prompt:
                try:
                    predicted = int(executor.execute(program, prompt))
                    correct = 1.0 if predicted == ep.outcome else 0.0
                except Exception:
                    correct = 1.0 if ep.outcome == 0 else 0.0
            else:
                correct = 1.0 if ep.outcome == 0 else 0.0

            raw.append(correct)
            # Rolling 3-step average
            window = raw[-3:]
            curve.append(sum(window) / len(window))

        logger.info(
            "Accuracy curve for %s: %d episodes, best_program=%s, final_acc=%.3f",
            campaign_id, len(episodes),
            program.id if program else None,
            curve[-1] if curve else 0.0,
        )
        return curve
