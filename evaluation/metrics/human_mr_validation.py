"""Human validation for Malicious Rate (MR).

Samples victim responses from a completed campaign, exports them for human
annotation (SERIOUS vs SLIGHT), and computes inter-rater agreement between
human annotators and the LLM judge (Cohen's kappa).

Usage:
    # Step 1: Export responses for human annotation
    python -m evaluation.metrics.human_mr_validation export \\
        --campaign-dir outputs/campaign/<campaign_id> \\
        --num-samples 100 \\
        --output human_mr_samples.csv

    # Step 2: After humans annotate, compute agreement
    python -m evaluation.metrics.human_mr_validate evaluate \\
        --annotations human_mr_annotated.csv \\
        --judge-labels human_mr_samples.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("human_mr_validation")


def export_samples(
    campaign_dir: str,
    num_samples: int = 100,
    output_path: str = "human_mr_samples.csv",
    seed: int = 42,
) -> None:
    """Export victim responses from a campaign for human MR annotation.

    Each row shows the original prompt, crafted attack prompt, and victim
    response. Annotators mark SERIOUS or SLIGHT.
    """
    campaign_path = Path(campaign_dir)
    eval_dir = campaign_path / "evaluation"

    # Try loading from harmonyx_asr details first, then malicious_rate.json
    responses: List[Dict[str, str]] = []

    # Source 1: HarmonyX ASR evaluation details
    report_path = eval_dir / "evaluation_report.json"
    if report_path.exists():
        with open(report_path) as f:
            report = json.load(f)
        details = report.get("harmonyx_asr", {}).get("details", [])
        for d in details:
            if d.get("victim_response") and str(d.get("victim_response", "")).strip():
                responses.append({
                    "goal_prompt": d.get("goal", ""),
                    "crafted_prompt": d.get("crafted_prompt", ""),
                    "victim_response": str(d.get("victim_response", "")),
                    "llm_judge_outcome": "ACCEPT" if d.get("outcome") == 0 else "REFUSE",
                    "technique": d.get("technique", ""),
                })

    # Source 2: Malicious rate results (SERIOUS/SLIGHT classifications)
    mr_path = campaign_path / "evaluation" / "malicious_rate.json"
    if mr_path.exists():
        with open(mr_path) as f:
            mr_data = json.load(f)
        for entry in mr_data:
            if entry.get("response", "").strip():
                responses.append({
                    "goal_prompt": entry.get("prompt", ""),
                    "crafted_prompt": entry.get("crafted_prompt", entry.get("prompt", "")),
                    "victim_response": entry.get("response", ""),
                    "llm_judge_outcome": entry.get("classification", "UNKNOWN"),
                    "technique": entry.get("technique", ""),
                })

    if not responses:
        logger.error(
            "No victim responses found in %s. "
            "Run HarmonyX ASR evaluation first.",
            campaign_dir,
        )
        sys.exit(1)

    rng = random.Random(seed)
    sampled = rng.sample(responses, min(num_samples, len(responses)))

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "item_id", "goal_prompt", "crafted_prompt",
            "victim_response", "technique",
            "llm_judge_outcome",
            "human_label",  # To be filled by annotator
            "annotator_id",  # Annotator identifier
        ])
        for i, r in enumerate(sampled):
            writer.writerow([
                i,
                r["goal_prompt"],
                r["crafted_prompt"],
                r["victim_response"],
                r["technique"],
                r["llm_judge_outcome"],
                "",  # human_label (to be filled)
                "",  # annotator_id (to be filled)
            ])

    logger.info(
        "Exported %d samples to %s",
        len(sampled), output_path,
    )
    logger.info("Instructions: Fill 'human_label' with SERIOUS or SLIGHT for each row.")
    logger.info("             Fill 'annotator_id' with a unique identifier per annotator.")


def evaluate_agreement(
    annotations_path: str,
    judge_labels_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Compute agreement between human annotators and LLM judge.

    Parameters
    ----------
    annotations_path : str
        CSV with columns: item_id, human_label, annotator_id, llm_judge_outcome.
    judge_labels_path : str, optional
        If provided, merges judge labels from the exported samples.

    Returns
    -------
    dict with keys ``"cohens_kappa"``, ``"agreement_pct"``,
    ``"n_items"``, ``"n_annotators"``, ``"confusion_matrix"``.
    """
    from evaluation.metrics.statistical_tests import cohens_kappa

    # Load annotations
    annotations: List[Dict[str, Any]] = []
    with open(annotations_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            human_label = row.get("human_label", "").strip().upper()
            annotator_id = row.get("annotator_id", "").strip()
            judge_label = row.get("llm_judge_outcome", "").strip().upper()
            if human_label and human_label in ("SERIOUS", "SLIGHT"):
                annotations.append({
                    "item_id": row.get("item_id", ""),
                    "human_label": human_label,
                    "annotator_id": annotator_id,
                    "llm_judge_outcome": judge_label,
                })

    if not annotations:
        logger.error("No valid annotations found. Check human_label values (SERIOUS/SLIGHT).")
        return {"error": "No valid annotations"}

    # Map labels to binary: SERIOUS=1, SLIGHT=0
    label_map = {"SERIOUS": 1, "SLIGHT": 0}

    # Per-annotator agreement with judge
    annotator_groups: Dict[str, Dict[str, int]] = {}
    for ann in annotations:
        aid = ann["annotator_id"] or "default"
        if aid not in annotator_groups:
            annotator_groups[aid] = {"human": [], "judge": []}
        h = label_map.get(ann["human_label"], -1)
        j = label_map.get(ann["llm_judge_outcome"], -1)
        if h != -1 and j != -1:
            annotator_groups[aid]["human"].append(h)
            annotator_groups[aid]["judge"].append(j)

    # Overall agreement (aggregate all annotator judgments)
    all_human = []
    all_judge = []
    for aid, groups in annotator_groups.items():
        all_human.extend(groups["human"])
        all_judge.extend(groups["judge"])

    if not all_human:
        return {"error": "No matching labels found"}

    kappa_result = cohens_kappa(all_human, all_judge)

    # Per-annotator breakdown
    per_annotator = {}
    for aid, groups in annotator_groups.items():
        if len(groups["human"]) > 0:
            ka = cohens_kappa(groups["human"], groups["judge"])
            per_annotator[aid] = {
                "n": len(groups["human"]),
                "kappa": ka["kappa"],
                "agreement_pct": ka["agreement_pct"],
            }

    # Confusion matrix
    cm = {"TP": 0, "FP": 0, "TN": 0, "FN": 0}
    for h, j in zip(all_human, all_judge):
        if h == 1 and j == 1:
            cm["TP"] += 1
        elif h == 0 and j == 1:
            cm["FP"] += 1
        elif h == 0 and j == 0:
            cm["TN"] += 1
        elif h == 1 and j == 0:
            cm["FN"] += 1

    # Compute Fleiss' kappa (multi-annotator agreement)
    fleiss_kappa = _fleiss_kappa(annotator_groups)

    result = {
        "n_total_annotations": len(all_human),
        "n_annotators": len(annotator_groups),
        "overall_kappa": kappa_result["kappa"],
        "overall_agreement_pct": kappa_result["agreement_pct"],
        "strength": kappa_result["strength"],
        "fleiss_kappa": fleiss_kappa,
        "confusion_matrix": cm,
        "per_annotator": per_annotator,
    }

    # Print summary
    print()
    print("=" * 60)
    print("HUMAN MR VALIDATION RESULTS")
    print("=" * 60)
    print(f"  Annotations:      {result['n_total_annotations']}")
    print(f"  Annotators:       {result['n_annotators']}")
    print(f"  Cohen's κ:        {result['overall_kappa']:.4f} ({result['strength']})")
    print(f"  Agreement %:      {result['overall_agreement_pct']:.1f}%")
    print(f"  Fleiss' κ:        {result['fleiss_kappa']:.4f}")
    print(f"  Confusion matrix: {cm}")
    if cm["TP"] + cm["FN"] > 0:
        print(f"  Sensitivity (TPR): {cm['TP'] / (cm['TP'] + cm['FN']):.3f}")
    if cm["TN"] + cm["FP"] > 0:
        print(f"  Specificity (TNR): {cm['TN'] / (cm['TN'] + cm['FP']):.3f}")
    print()
    for aid, detail in per_annotator.items():
        print(f"  Annotator '{aid}': n={detail['n']}, κ={detail['kappa']:.4f}, "
              f"agree={detail['agreement_pct']:.1f}%")
    print("=" * 60)

    return result


def _fleiss_kappa(annotator_groups: Dict[str, Dict[str, List[int]]]) -> float:
    """Compute Fleiss' kappa for multi-annotator agreement.

    Simplified: treats SERIOUS=1, SLIGHT=0 as binary categories.
    """
    # Build item-level matrix: for each item, count of SERIOUS and SLIGHT
    item_ratings: Dict[str, List[int]] = {}
    for aid, groups in annotator_groups.items():
        for i, (h, j) in enumerate(zip(groups["human"], groups["judge"])):
            item_id = f"item_{i}"
            if item_id not in item_ratings:
                item_ratings[item_id] = [0, 0]  # [SERIOUS, SLIGHT]
            if h == 1:
                item_ratings[item_id][0] += 1
            else:
                item_ratings[item_id][1] += 1
            if j == 1:
                item_ratings[item_id][0] += 1
            else:
                item_ratings[item_id][1] += 1

    items = list(item_ratings.values())
    n_items = len(items)
    if n_items == 0:
        return 0.0

    n_raters = max(sum(r) for r in items)
    if n_raters <= 1:
        return 0.0

    n_categories = 2

    # Observed agreement
    p_i = []
    for ratings in items:
        total = sum(ratings)
        if total <= 1:
            continue
        agreement = sum(r * (r - 1) for r in ratings) / (total * (total - 1))
        p_i.append(agreement)

    if not p_i:
        return 0.0
    p_bar = sum(p_i) / len(p_i)

    # Expected agreement
    p_j = [sum(items[i][j] for i in range(n_items)) / (n_items * n_raters) for j in range(n_categories)]
    p_e = sum(pj ** 2 for pj in p_j)

    if abs(p_e - 1.0) < 1e-10:
        return 1.0
    return (p_bar - p_e) / (1.0 - p_e)


def main():
    parser = argparse.ArgumentParser(
        description="Human validation for Malicious Rate (MR)",
    )
    subparsers = parser.add_subparsers(dest="command")

    # Export
    export_parser = subparsers.add_parser("export", help="Export responses for annotation")
    export_parser.add_argument("--campaign-dir", required=True)
    export_parser.add_argument("--num-samples", type=int, default=100)
    export_parser.add_argument("--output", default="human_mr_samples.csv")

    # Evaluate
    eval_parser = subparsers.add_parser("evaluate", help="Compute inter-rater agreement")
    eval_parser.add_argument("--annotations", required=True)
    eval_parser.add_argument("--judge-labels", default=None)

    args = parser.parse_args()

    if args.command == "export":
        export_samples(args.campaign_dir, args.num_samples, args.output)
    elif args.command == "evaluate":
        evaluate_agreement(args.annotations, args.judge_labels)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
