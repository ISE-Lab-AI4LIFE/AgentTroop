from __future__ import annotations

import logging
from typing import Any, Optional

from evaluation.metrics.explanation_score import (
    AnnotatorRating,
    ExplanationScore,
    ExplanationScoreMetric,
)

logger = logging.getLogger(__name__)


class RQ2Evaluator:
    """RQ2: Can the system discover novel defense components (human evaluation)?

    Exports discovered programs and theories for annotation, then computes
    Likert scores and Fleiss' Kappa.
    """

    def __init__(self) -> None:
        self._metric = ExplanationScoreMetric()

    def export_for_annotation(
        self,
        programs_data: list[dict],
        output_path: str,
    ) -> str:
        return self._metric.export_for_annotation(programs_data, output_path)

    def evaluate(
        self,
        annotation_path: str,
        program_id: str,
        theory_pattern: str = "",
    ) -> dict:
        score = self._metric.compute_from_file(annotation_path, program_id, theory_pattern)
        result = score.to_dict()
        result["rq"] = "RQ2"
        logger.info(
            "RQ2: program=%s overall_mean=%.2f fleiss_kappa=%.2f",
            program_id, score.overall_mean, score.fleiss_kappa,
        )
        return result
