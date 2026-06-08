"""Optional MLflow integration for experiment tracking.

Logs parameters, metrics, and artifacts for each HARMONY-X campaign.
MLflow is optional; if not installed, these calls are no-ops.

Usage:
    from experiments.mlflow_tracking import log_campaign

    log_campaign(
        campaign_id="campaign_001",
        params={"max_iterations": 50, "model": "ToyVictim"},
        metrics={"accuracy": 0.92, "interventions": 45},
    )
"""

import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

try:
    import mlflow

    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False


def _ensure_run() -> bool:
    if not MLFLOW_AVAILABLE:
        return False
    if mlflow.active_run() is None:
        try:
            mlflow.start_run()
        except Exception:
            return False
    return True


def log_campaign(
    campaign_id: str,
    params: Optional[Dict[str, Any]] = None,
    metrics: Optional[Dict[str, float]] = None,
    artifacts: Optional[Dict[str, str]] = None,
) -> bool:
    """Log a campaign's parameters, metrics, and artifacts to MLflow.

    Returns True if logging succeeded, False if MLflow is unavailable.
    """
    if not _ensure_run():
        logger.debug("MLflow not available; skipping logging for %s", campaign_id)
        return False

    try:
        mlflow.set_tag("campaign_id", campaign_id)
        if params:
            mlflow.log_params(params)
        if metrics:
            mlflow.log_metrics(metrics)
        if artifacts:
            for name, path in artifacts.items():
                mlflow.log_artifact(path, artifact_path=name)
        return True
    except Exception as exc:
        logger.warning("MLflow logging failed: %s", exc)
        return False


def log_campaign_result(result: Dict[str, Any], campaign_id: str) -> bool:
    """Convenience: log the result dict from Orchestrator.run()."""
    params = {
        "campaign_id": campaign_id,
        "max_interventions": result.get("total_interventions", 0),
    }
    metrics = {
        "accuracy": float(result.get("best_accuracy", 0.0)),
        "interventions": float(result.get("total_interventions", 0)),
        "iterations": float(result.get("total_iterations", 0)),
    }
    return log_campaign(
        campaign_id=campaign_id,
        params=params,
        metrics=metrics,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logged = log_campaign(
        campaign_id="test_campaign",
        params={"max_iterations": 50, "model": "ToyVictim"},
        metrics={"accuracy": 0.92, "interventions": 45},
    )
    print(f"MLflow logging: {'SUCCESS' if logged else 'SKIPPED (not available)'}")
