#!/usr/bin/env python3
"""
Holdout evaluation: test the learned program against paraphrased prompts
with ZERO lexical overlap to training data.

Usage:
    python3 police_validation/evaluate_holdout.py \
        --program llama3_1_8b/outputs/final_program.json \
        --holdout police_validation/holdout_prompts.csv

    # Evaluate all experiments in results/ (loads program from defense store)
    python3 police_validation/evaluate_holdout.py --all
"""

import argparse
import csv
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("holdout_eval")

RESULTS_DIR = Path(__file__).resolve().parent / "results"


def load_holdout_csv(path: str) -> List[Tuple[str, int]]:
    """Load holdout prompts CSV. Expected columns: prompt, label (0=ACCEPT,1=REFUSE)."""
    pairs: List[Tuple[str, int]] = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            logger.error("Empty CSV: %s", path)
            return pairs
        prompt_col = "prompt"
        label_col = "label"
        for col in reader.fieldnames:
            if col.strip().lower() == "prompt":
                prompt_col = col
            if col.strip().lower() == "label":
                label_col = col
        for row in reader:
            prompt = row.get(prompt_col, "").strip()
            label_str = row.get(label_col, "").strip()
            if prompt and label_str:
                pairs.append((prompt, int(label_str)))
    logger.info("Loaded %d holdout prompts from %s", len(pairs), path)
    return pairs


def load_program_from_json(path: str) -> Optional[Any]:
    """Load a Program from a saved JSON file."""
    from core.program import Program

    with open(path) as f:
        data = json.load(f)
    if isinstance(data, dict) and "root" in data:
        program = Program.from_dict(data)
        logger.info("Loaded program from %s (complexity=%d)", path, program.complexity())
        return program
    logger.error("Invalid program JSON: %s", path)
    return None


def _keyword_baseline(prompt: str) -> int:
    """Simple keyword classifier: return 1 (REFUSE) if prompt contains
    refusal-triggering keywords, else 0 (ACCEPT).

    This provides a lower-bound comparison for the learned program.
    """
    refuse_keywords = [
        "how to make", "instructions for", "walk me through",
        "teach me how", "step-by-step", "guide to",
        "give me a", "tell me how", "describe how to",
        "show me how", "explain how to", "provide instructions",
        "write instructions", "producing", "manufacturing",
        "synthesize", "smuggling", "embezzling",
        "bypass", "illegal", "criminal",
        "unauthorized", "counterfeit", "black market",
        "steal", "break into", "dump",
    ]
    prompt_lower = prompt.lower()
    for kw in refuse_keywords:
        if kw in prompt_lower:
            return 1
    return 0


def evaluate_program(
    program: Any,
    holdout_pairs: List[Tuple[str, int]],
    executor: Any,
) -> Dict[str, float]:
    """Evaluate a program against holdout prompts.

    Returns accuracy, precision, recall, F1, and per-prompt predictions.
    """
    from core.primitive import PrimitiveRegistry

    y_true: List[int] = []
    y_pred: List[int] = []
    details: List[Dict[str, Any]] = []

    for prompt, label in holdout_pairs:
        pred = int(executor.execute(program, prompt))
        y_true.append(label)
        y_pred.append(pred)
        details.append({
            "prompt": prompt[:80],
            "label": label,
            "predicted": pred,
            "correct": pred == label,
        })

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    accuracy = float(np.mean(y_true == y_pred))
    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "n_total": len(holdout_pairs),
        "n_refuse": tp + fn,
        "n_accept": tn + fp,
        "details": details,
    }


def run_keyword_baseline(holdout_pairs: List[Tuple[str, int]]) -> Dict[str, float]:
    """Evaluate keyword baseline against holdout."""
    y_true = np.array([label for _, label in holdout_pairs])
    y_pred = np.array([_keyword_baseline(prompt) for prompt, _ in holdout_pairs])

    accuracy = float(np.mean(y_true == y_pred))
    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "n_total": len(holdout_pairs),
    }


def print_results(name: str, results: Dict[str, float], keyword_results: Optional[Dict[str, float]] = None):
    """Pretty-print evaluation results."""
    print(f"\n{'=' * 60}")
    print(f"  {name}")
    print(f"{'=' * 60}")
    print(f"  Accuracy:  {results['accuracy']:.2%}  ({results['tp']+results['tn']}/{results['n_total']})")
    print(f"  Precision: {results['precision']:.2%}  ({results['tp']}/{results['tp']+results['fp']})")
    print(f"  Recall:    {results['recall']:.2%}  ({results['tp']}/{results['tp']+results['fn']})")
    print(f"  F1 Score:  {results['f1']:.2%}")
    print(f"  Confusion:")
    print(f"    TP={results['tp']}  FP={results['fp']}")
    print(f"    FN={results['fn']}  TN={results['tn']}")

    if keyword_results:
        print(f"\n  --- Vs. Keyword Baseline ---")
        print(f"  Keyword accuracy:  {keyword_results['accuracy']:.2%}")
        print(f"  Keyword F1:        {keyword_results['f1']:.2%}")
        delta_acc = results['accuracy'] - keyword_results['accuracy']
        delta_f1 = results['f1'] - keyword_results['f1']
        if delta_acc > 0:
            print(f"  ✓ Learned program beats keyword by {delta_acc:.1%} accuracy")
        elif delta_acc < 0:
            print(f"  ✗ Keyword beats learned program by {-delta_acc:.1%} accuracy")
        else:
            print(f"  ∼ Tie with keyword baseline")

    # Show failures
    failures = [d for d in results.get("details", []) if not d["correct"]]
    if failures:
        print(f"\n  --- Failures ({len(failures)}) ---")
        for d in failures[:10]:
            pred_label = "REFUSE" if d["predicted"] else "ACCEPT"
            true_label = "REFUSE" if d["label"] else "ACCEPT"
            print(f"  Pred={pred_label:<6} True={true_label:<6} | {d['prompt'][:60]}")
        if len(failures) > 10:
            print(f"  ... and {len(failures) - 10} more")


def main():
    parser = argparse.ArgumentParser(description="Holdout evaluation")
    parser.add_argument("--program", default=None,
                        help="Path to final_program.json")
    parser.add_argument("--holdout", default=None,
                        help="Path to holdout_prompts.csv")
    parser.add_argument("--all", action="store_true",
                        help="Evaluate all experiments")
    args = parser.parse_args()

    from core.executor import ProgramExecutor
    from core.primitive import PrimitiveRegistry

    # Determine holdout path
    holdout_path = args.holdout or str(Path(__file__).resolve().parent / "holdout_prompts.csv")
    if not Path(holdout_path).exists():
        logger.error("Holdout CSV not found: %s", holdout_path)
        logger.error("Generate it with: python3 run_validation.py --gen-holdout")
        sys.exit(1)

    holdout_pairs = load_holdout_csv(holdout_path)

    # Keyword baseline (always run since it's cheap)
    keyword_results = run_keyword_baseline(holdout_pairs)
    print_results("KEYWORD BASELINE", keyword_results)

    if args.all:
        # Evaluate all experiments
        program_paths = list((Path(_PROJECT_ROOT) / "llama3_1_8b" / "outputs").glob("*/final_program.json"))
        exp_paths = list(RESULTS_DIR.glob("*_candidates.json"))

        if not program_paths and not exp_paths:
            logger.error("No experiment outputs found")
            sys.exit(1)

        for exp_path in exp_paths:
            exp_name = exp_path.stem.replace("_candidates", "")
            try:
                with open(exp_path) as f:
                    data = json.load(f)
                best_id = data.get("result", {}).get("best_program_id")
                if not best_id:
                    logger.warning("No best program in %s", exp_path)
                    continue

                # Try to load from defense store
                from knowledge.defense_store import DefenseProgramStore
                neo4j_uri = "bolt://localhost:7687"
                store = DefenseProgramStore(uri=neo4j_uri, user="neo4j", password="password")
                record = store.get(best_id)
                store.close()

                if record is None:
                    logger.warning("Program %s not found in defense store", best_id)
                    continue

                program = record.program if hasattr(record, "program") else record
                executor = ProgramExecutor(registry=PrimitiveRegistry())

                results = evaluate_program(program, holdout_pairs, executor)
                print_results(f"LEARNED: {exp_name}", results, keyword_results)
            except Exception as e:
                logger.error("Failed to evaluate %s: %s", exp_path, e)

    elif args.program:
        # Single program evaluation
        program = load_program_from_json(args.program)
        if program is None:
            sys.exit(1)

        executor = ProgramExecutor(registry=PrimitiveRegistry())
        results = evaluate_program(program, holdout_pairs, executor)
        print_results(f"LEARNED: {Path(args.program).parent.name}", results, keyword_results)

        # Save detailed results
        output_path = Path(args.program).parent / "holdout_evaluation.json"
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        logger.info("Detailed results saved to %s", output_path)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
