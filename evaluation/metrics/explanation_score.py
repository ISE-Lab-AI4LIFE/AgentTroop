from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class AnnotatorRating:
    annotator_id: str
    consistency: int
    clarity: int
    generality: int
    notes: str = ""

    def to_dict(self) -> dict:
        return {
            "annotator_id": self.annotator_id,
            "consistency": self.consistency,
            "clarity": self.clarity,
            "generality": self.generality,
            "notes": self.notes,
        }


@dataclass
class ExplanationScore:
    program_id: str
    theory_pattern: str
    ratings: list[AnnotatorRating] = field(default_factory=list)
    mean_consistency: float = 0.0
    mean_clarity: float = 0.0
    mean_generality: float = 0.0
    overall_mean: float = 0.0
    fleiss_kappa: float = 0.0

    def to_dict(self) -> dict:
        return {
            "program_id": self.program_id,
            "theory_pattern": self.theory_pattern,
            "ratings": [r.to_dict() for r in self.ratings],
            "mean_consistency": self.mean_consistency,
            "mean_clarity": self.mean_clarity,
            "mean_generality": self.mean_generality,
            "overall_mean": self.overall_mean,
            "fleiss_kappa": self.fleiss_kappa,
        }


class ExplanationScoreMetric:
    """RQ2: Human evaluation of discovered programs and theories.

    Exports programs/theories to a JSON file for annotation, then computes
    mean Likert scores and Fleiss' Kappa from annotator ratings.
    """

    def export_for_annotation(
        self,
        programs: list[dict],
        output_path: str,
    ) -> str:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(programs, f, indent=2, default=str)
        logger.info("Exported %d programs to %s for annotation", len(programs), output_path)
        return output_path

    def compute_scores(
        self,
        ratings: list[AnnotatorRating],
        program_id: str,
        theory_pattern: str = "",
    ) -> ExplanationScore:
        if not ratings:
            return ExplanationScore(
                program_id=program_id, theory_pattern=theory_pattern,
            )

        n = len(ratings)
        mean_c = sum(r.consistency for r in ratings) / n
        mean_cl = sum(r.clarity for r in ratings) / n
        mean_g = sum(r.generality for r in ratings) / n

        score = ExplanationScore(
            program_id=program_id,
            theory_pattern=theory_pattern,
            ratings=ratings,
            mean_consistency=mean_c,
            mean_clarity=mean_cl,
            mean_generality=mean_g,
            overall_mean=(mean_c + mean_cl + mean_g) / 3,
        )
        score.fleiss_kappa = self._fleiss_kappa(ratings)
        return score

    def compute_from_file(
        self,
        annotation_path: str,
        program_id: str,
        theory_pattern: str = "",
    ) -> ExplanationScore:
        with open(annotation_path) as f:
            data = json.load(f)
        ratings = [
            AnnotatorRating(
                annotator_id=r["annotator_id"],
                consistency=r["consistency"],
                clarity=r["clarity"],
                generality=r["generality"],
                notes=r.get("notes", ""),
            )
            for r in data
        ]
        return self.compute_scores(ratings, program_id, theory_pattern)

    def _fleiss_kappa(self, ratings: list[AnnotatorRating]) -> float:
        if len(ratings) < 2:
            return 0.0
        categories = [1, 2, 3, 4, 5]
        n_subjects = 1
        n_raters = len(ratings)
        N = n_subjects * n_raters

        agreements = []
        for cat in categories:
            count = sum(1 for r in ratings if r.consistency == cat)
            count += sum(1 for r in ratings if r.clarity == cat)
            count += sum(1 for r in ratings if r.generality == cat)
            agreements.append(count)

        p_i = [a / N for a in agreements]
        P_bar = sum(p * p for p in p_i)

        P_e = sum(p_i)
        K = (P_bar - P_e) / (1 - P_e) if P_e < 1 else 0.0
        return max(0.0, K)
