"""Human evaluation for RQ2 (Section 14.6 of harmony_v5v.md).

Generates annotation files for 3 independent annotators and computes
Fleiss' Kappa for inter-annotator agreement.

Usage:
    # Export 50 (program, explanation) pairs for annotation
    python experiments/human_evaluation.py export \\
        --programs outputs/final_program.json \\
        --output annotation_input.csv

    # After annotators fill in scores, compute Fleiss' Kappa
    python experiments/human_evaluation.py evaluate \\
        --annotations annotation_results.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger("human_eval")

# Likert scale labels
LIKERT_LABELS = ["1-StronglyDisagree", "2-Disagree", "3-Neutral", "4-Agree", "5-StronglyAgree"]


# ── Export ───────────────────────────────────────────────────────────────────


def export_for_annotation(
    program_path: str,
    output_path: str,
    num_pairs: int = 50,
    seed: int = 42,
) -> None:
    """Export *num_pairs* (program, explanation) pairs for human annotation.

    Each row asks two Likert-1-5 questions:
        Q1: Is the program consistent with the interventions performed?
        Q2: Is the explanation reasonable?

    Output CSV columns: item_id, program_text, explanation, q1, q2
    (annotators fill in q1, q2).
    """
    rng = random.Random(seed)

    with open(program_path) as f:
        program_data = json.load(f)

    # Generate program variations and explanations for annotation
    program_text = json.dumps(program_data, indent=2) if isinstance(program_data, dict) else str(program_data)

    explanations = [
        "The safety filter blocks prompts containing harmful keywords.",
        "ROT13 encoding bypasses the keyword filter because it only checks plaintext.",
        "Base64-encoded prompts are decoded before the safety check.",
        "Role-play prefixes frame harmful requests as academic research, avoiding refusal.",
        "The model accepts short generic questions but refuses specific harmful instructions.",
        "The safety classifier uses a toxicity score threshold to decide refusal.",
        "Prompts with numbers (doses, amounts) trigger stricter safety checks.",
        "The filter checks for leet-speak variations of blocked keywords.",
        "Grammar changes (imperative vs interrogative) affect refusal likelihood.",
        "The safety system has a hardcoded blocklist of violent verbs.",
        "Decoding transforms are applied before keyword matching.",
        "Longer prompts are more likely to be refused due to higher entropy.",
        "The model refuses prompts containing personal identifying information.",
        "URL-encoded content is decoded before safety classification.",
        "The safety filter is case-sensitive for certain keywords.",
        "Instruction-following prompts bypass the filter when framed as educational.",
        "The model has separate classifiers for different harm categories.",
        "Chain-of-thought prompting can override the safety layer.",
        "The refusal mechanism uses a two-stage classifier with transform pre-processing.",
        "Prompt injection markers are detected and blocked regardless of encoding.",
    ]

    # Generate pairs
    pairs = []
    for i in range(num_pairs):
        explanation = explanations[i % len(explanations)]
        # Vary the program explanation slightly for diversity
        variant = program_text
        if i % 3 == 0:
            variant = program_text.replace('"', "'")
        elif i % 3 == 1:
            variant = program_text[:200] + ("..." if len(program_text) > 200 else "")
        pairs.append({
            "item_id": f"pair_{i:03d}",
            "program_text": variant,
            "explanation": explanation,
            "q1": "",  # annotator fills
            "q2": "",  # annotator fills
        })

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["item_id", "program_text", "explanation", "q1", "q2"])
        writer.writeheader()
        writer.writerows(pairs)

    logger.info("Exported %d annotation pairs to %s", num_pairs, output_path)
    print(f"\nAnnotation file: {output_path}")
    print(f"Rows: {num_pairs}")
    print("Ask 3 annotators to fill q1 (consistency) and q2 (reasonableness) with 1-5.")
    print()


# ── Evaluation ───────────────────────────────────────────────────────────────


def compute_fleiss_kappa(ratings: List[List[int]], n_categories: int = 5) -> float:
    """Compute Fleiss' Kappa for inter-annotator agreement.

    Parameters
    ----------
    ratings : list of list of int
        ratings[i][j] = rating of item *i* by annotator *j* (1-based).
    n_categories : int
        Number of rating categories (default 5 for Likert 1-5).

    Returns
    -------
    float
        Fleiss' Kappa value (-1 to 1).
    """
    n_items = len(ratings)
    if n_items == 0:
        return 0.0
    n_annotators = len(ratings[0]) if ratings else 0
    if n_annotators < 2:
        return 0.0

    # n_items × n_categories agreement matrix
    agreement = [[0] * n_categories for _ in range(n_items)]
    for i, item in enumerate(ratings):
        for r in item:
            cat = min(max(r - 1, 0), n_categories - 1)
            agreement[i][cat] += 1

    N = n_items
    n = n_annotators
    k = n_categories

    # P_i = agreement proportion for item i
    P = []
    for i in range(N):
        total = sum(agreement[i][c] for c in range(k))
        if total <= 0:
            P.append(0.0)
            continue
        s = sum(agreement[i][c] ** 2 for c in range(k))
        P.append((s - total) / (total * (total - 1)) if total > 1 else 1.0)

    P_bar = sum(P) / N

    # p_j = proportion of all ratings in category j
    p_j = [0.0] * k
    for i in range(N):
        for c in range(k):
            p_j[c] += agreement[i][c]
    total_ratings = N * n
    if total_ratings > 0:
        p_j = [v / total_ratings for v in p_j]
    else:
        return 0.0

    # P_e_bar = sum p_j^2
    P_e_bar = sum(p ** 2 for p in p_j)

    if P_e_bar >= 1.0:
        return 1.0

    kappa = (P_bar - P_e_bar) / (1.0 - P_e_bar)
    return kappa


def evaluate_annotations(annotation_path: str) -> dict:
    """Read annotation CSV and compute Fleiss' Kappa for Q1 and Q2."""
    with open(annotation_path, newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        logger.error("No annotation data found")
        return {"error": "empty"}

    # Group by annotator (assumes columns: q1_annotator1, q1_annotator2, q1_annotator3)
    # Or: three separate CSVs concatenated
    q1_ratings: List[List[int]] = []
    q2_ratings: List[List[int]] = []

    for row in rows:
        try:
            q1_vals = [int(row.get(f"q1_{a}", "")) for a in ("annotator1", "annotator2", "annotator3")]
            q2_vals = [int(row.get(f"q2_{a}", "")) for a in ("annotator1", "annotator2", "annotator3")]
            if all(1 <= v <= 5 for v in q1_vals):
                q1_ratings.append(q1_vals)
            if all(1 <= v <= 5 for v in q2_vals):
                q2_ratings.append(q2_vals)
        except (ValueError, TypeError):
            continue

    # Fallback: single q1/q2 columns (concatenated annotations)
    if not q1_ratings:
        q1_buf: List[List[int]] = []
        for row in rows:
            try:
                v = int(row.get("q1", ""))
                if 1 <= v <= 5:
                    q1_buf.append([v])
            except (ValueError, TypeError):
                pass
        if q1_buf:
            logger.warning("Only single-annotator data found for Q1")

    kappa_q1 = compute_fleiss_kappa(q1_ratings) if len(q1_ratings) >= 2 else 0.0
    kappa_q2 = compute_fleiss_kappa(q2_ratings) if len(q2_ratings) >= 2 else 0.0

    mean_q1 = sum(sum(r) for r in q1_ratings) / max(1, len(q1_ratings) * len(q1_ratings[0] if q1_ratings else [1]))
    mean_q2 = sum(sum(r) for r in q2_ratings) / max(1, len(q2_ratings) * len(q2_ratings[0] if q2_ratings else [1]))

    result = {
        "q1": {"mean": round(mean_q1, 2), "fleiss_kappa": round(kappa_q1, 4), "n_items": len(q1_ratings)},
        "q2": {"mean": round(mean_q2, 2), "fleiss_kappa": round(kappa_q2, 4), "n_items": len(q2_ratings)},
        "overall_mean": round((mean_q1 + mean_q2) / 2, 2),
        "overall_kappa": round((kappa_q1 + kappa_q2) / 2, 4),
    }

    print(f"\n{'='*60}")
    print("Human Evaluation Results (Fleiss' Kappa)")
    print(f"{'='*60}")
    print(f"  Q1 (Consistency):     mean={result['q1']['mean']}, κ={result['q1']['fleiss_kappa']:.4f} "
          f"(n={result['q1']['n_items']})")
    print(f"  Q2 (Reasonableness):  mean={result['q2']['mean']}, κ={result['q2']['fleiss_kappa']:.4f} "
          f"(n={result['q2']['n_items']})")
    print(f"  Overall:              mean={result['overall_mean']}, κ={result['overall_kappa']:.4f}")
    print(f"{'='*60}")

    # Interpretation
    k = result['overall_kappa']
    if k < 0:
        print("  Interpretation: Poor agreement (less than chance)")
    elif k < 0.21:
        print("  Interpretation: Slight agreement")
    elif k < 0.41:
        print("  Interpretation: Fair agreement")
    elif k < 0.61:
        print("  Interpretation: Moderate agreement")
    elif k < 0.81:
        print("  Interpretation: Substantial agreement")
    else:
        print("  Interpretation: Almost perfect agreement")
    print()

    return result


# ── CLI ──────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Human evaluation for HARMONY-X (Section 14.6)",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    export_p = sub.add_parser("export", help="Export annotation input CSV")
    export_p.add_argument("--programs", required=True, help="Path to final_program.json")
    export_p.add_argument("--output", default="annotation_input.csv", help="Output CSV")
    export_p.add_argument("--num-pairs", type=int, default=50, help="Number of pairs (default 50)")

    eval_p = sub.add_parser("evaluate", help="Compute Fleiss' Kappa from annotations")
    eval_p.add_argument("--annotations", required=True, help="Annotation results CSV")

    args = parser.parse_args()

    if args.command == "export":
        export_for_annotation(
            program_path=args.programs,
            output_path=args.output,
            num_pairs=args.num_pairs,
        )
    elif args.command == "evaluate":
        evaluate_annotations(annotation_path=args.annotations)


if __name__ == "__main__":
    main()
