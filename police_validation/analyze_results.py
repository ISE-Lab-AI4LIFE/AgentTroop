#!/usr/bin/env python3
"""
Post-experiment analysis: read candidate snapshots and produce summary tables.

Usage:
    python3 police_validation/analyze_results.py results/structural_candidates.json
    python3 police_validation/analyze_results.py --all  # analyze all results in results/
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

RESULTS_DIR = Path(__file__).resolve().parent / "results"


def load_snapshots(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def compute_type_progression(snapshots: list) -> list:
    """Convert raw snapshots into a clean type-progression table."""
    rows = []
    for s in snapshots:
        if s["event"] not in ("post_update", "run_end", "run_start"):
            continue
        row = {
            "cycle": s["cycle"],
            "iteration": s["iteration"],
            "event": s["event"],
            "n_candidates": s["n_candidates"],
            "entropy": round(s["entropy"], 4),
            "type_counts": dict(s["type_counts"]),
            "type_posterior": {k: round(v, 6) for k, v in s["type_posterior"].items()},
            "best_type": s["best"]["predicate_type"] if s["best"] else "none",
            "best_posterior": round(s["best"]["posterior"], 6) if s["best"] else 0,
            "best_holdout": round(s["best"]["holdout_accuracy"], 4) if s["best"] and s["best"].get("holdout_accuracy") else 0,
        }
        rows.append(row)
    return rows


def print_progression_table(rows: list, title: str = "Candidate Type Progression"):
    """Print a clean table of type posterior evolution."""
    # Collect all types seen across all snapshots
    all_types = set()
    for r in rows:
        all_types.update(r["type_posterior"].keys())
    all_types = sorted(all_types)

    print(f"\n{'=' * 80}")
    print(f"  {title}")
    print(f"{'=' * 80}")

    # Header
    header = f"{'Cycle':>5} | {'Event':<14} | {'N':>3} | {'Entropy':>8}"
    for t in all_types:
        header += f" | {t[:8]:>8}"
    header += " | Best"
    print(header)
    print("-" * len(header))

    for r in rows:
        line = f"{r['cycle']:5} | {r['event']:<14} | {r['n_candidates']:3} | {r['entropy']:8.4f}"
        for t in all_types:
            val = r["type_posterior"].get(t, 0)
            line += f" | {val:8.6f}"
        bt = r["best_type"]
        bp = r["best_posterior"]
        line += f" | {bt[:8]} ({bp:.3f})"
        print(line)

    # Final row: convergence check
    if rows:
        last = rows[-1]
        kw = last["type_posterior"].get("keyword", 0)
        st = last["type_posterior"].get("structural", 0)
        sem = last["type_posterior"].get("semantic", 0)
        tfm = last["type_posterior"].get("transform", 0)
        combined_st = st + sem + tfm
        winner = "keyword" if kw > combined_st else ("structural/semantic" if combined_st > kw else "tie")

        print(f"\n  Final state:")
        print(f"    Candidates: {last['n_candidates']}")
        print(f"    Entropy: {last['entropy']:.4f}")
        print(f"    Keyword posterior: {kw:.4%}")
        print(f"    Structural posterior: {st:.4%}")
        print(f"    Semantic posterior: {sem:.4%}")
        print(f"    Transform posterior: {tfm:.4%}")
        print(f"    Best: {last['best_type']} (post={last['best_posterior']:.4f}, holdout={last['best_holdout']:.2%})")
        print(f"    Winner: {winner}")


def compute_candidate_lifetimes(snapshots: list) -> list:
    """Track when each candidate appeared and its final posterior."""
    candidates = {}
    for s in snapshots:
        for c in s.get("candidates", []):
            pid = c["program_id"]
            if pid not in candidates:
                candidates[pid] = {
                    "program_id": pid,
                    "predicate_type": c["predicate_type"],
                    "source": c["source"],
                    "complexity": c["complexity"],
                    "first_seen": (s["cycle"], s["iteration"]),
                    "last_seen": (s["cycle"], s["iteration"]),
                    "accuracies": [],
                    "posteriors": [],
                    "holdouts": [],
                }
            candidates[pid]["last_seen"] = (s["cycle"], s["iteration"])
            candidates[pid]["accuracies"].append(c["accuracy"])
            candidates[pid]["posteriors"].append(c["posterior"])
            if c["holdout_accuracy"] and c["holdout_accuracy"] > 0:
                candidates[pid]["holdouts"].append(c["holdout_accuracy"])

    return sorted(candidates.values(), key=lambda x: -max(x["posteriors"]))


def print_candidate_table(candidates: list, top_n: int = 15):
    """Print candidate details sorted by max posterior."""
    print(f"\n  {'=' * 70}")
    print(f"  Candidate Detail (top {top_n} by posterior)")
    print(f"  {'=' * 70}")
    print(f"  {'ID':<20} {'Type':<14} {'Src':<14} {'Cplx':>4} {'Accuracy':>9} {'Max Post':>9} {'Holdouts':>8}")
    print(f"  {'-' * 78}")

    for c in candidates[:top_n]:
        max_post = max(c["posteriors"]) if c["posteriors"] else 0
        avg_holdout = np.mean(c["holdouts"]) if c["holdouts"] else 0
        acc = c["accuracies"][-1] if c["accuracies"] else 0
        print(f"  {c['program_id']:<20} {c['predicate_type']:<14} {c['source']:<14} "
              f"{c['complexity']:4} {acc:9.4f} {max_post:9.6f} {avg_holdout:8.4f}")


def compute_convergence_metrics(snapshots: list) -> dict:
    """Compute when (and if) the system converged."""
    metrics = {
        "converged": False,
        "convergence_cycle": None,
        "stable_candidate": None,
        "stable_from_cycle": None,
        "entropy_trajectory": [],
    }

    best_trace = []
    for s in snapshots:
        if s["best"]:
            metrics["entropy_trajectory"].append((s["cycle"], s["entropy"]))
            best_trace.append((s["cycle"], s["best"]["program_id"], s["best"]["predicate_type"]))

    # Check for stability (same best candidate for 3+ consecutive snapshots)
    if len(best_trace) >= 3:
        for i in range(len(best_trace) - 2):
            if best_trace[i][1] == best_trace[i + 1][1] == best_trace[i + 2][1]:
                metrics["converged"] = True
                metrics["convergence_cycle"] = best_trace[i][0]
                metrics["stable_candidate"] = best_trace[i][1]
                metrics["stable_type"] = best_trace[i][2]
                metrics["stable_from_cycle"] = best_trace[i][0]
                break

    # Final entropy
    if snapshots:
        metrics["final_entropy"] = snapshots[-1]["entropy"]
        metrics["final_n_candidates"] = snapshots[-1]["n_candidates"]

    return metrics


def print_convergence_report(metrics: dict):
    """Print convergence analysis."""
    print(f"\n  {'=' * 40}")
    print(f"  Convergence Report")
    print(f"  {'=' * 40}")
    print(f"    Converged: {metrics.get('converged', False)}")
    if metrics.get("converged"):
        print(f"    At cycle: {metrics['convergence_cycle']}")
        print(f"    Stable candidate: {metrics['stable_candidate']} ({metrics.get('stable_type', '?')})")
    else:
        print(f"    Did NOT converge within tracked cycles")
    print(f"    Final entropy: {metrics.get('final_entropy', 'N/A')}")
    print(f"    Final candidate count: {metrics.get('final_n_candidates', 'N/A')}")


def analyze(path: str):
    """Run full analysis on a single result file."""
    print(f"\nAnalyzing: {path}")
    data = load_snapshots(path)

    exp_name = data.get("experiment", Path(path).stem)
    snapshots = data.get("snapshots", [])
    result = data.get("result", {})

    print(f"  Experiment: {exp_name}")
    print(f"  Timestamp: {data.get('timestamp', '?')}")
    print(f"  Elapsed: {data.get('elapsed_seconds', 0):.1f}s")
    print(f"  Snapshots: {len(snapshots)}")
    print(f"  Final result: converged={result.get('converged')}, "
          f"best={result.get('best_predicate_type')}, "
          f"acc={result.get('accuracy', 0):.3f}")

    # Progression table
    rows = compute_type_progression(snapshots)
    print_progression_table(rows, title=f"Experiment: {exp_name}")

    # Candidate detail
    candidates = compute_candidate_lifetimes(snapshots)
    print_candidate_table(candidates)

    # Convergence
    metrics = compute_convergence_metrics(snapshots)
    print_convergence_report(metrics)

    # Final verdict
    last_snapshot = snapshots[-1] if snapshots else {}
    type_post = last_snapshot.get("type_posterior", {})
    kw = type_post.get("keyword", 0)
    combined_non_kw = sum(v for k, v in type_post.items() if k != "keyword")

    print(f"\n  {'=' * 40}")
    print(f"  VERDICT")
    print(f"  {'=' * 40}")
    if combined_non_kw > kw:
        print(f"  ✓ PASS: Non-keyword posterior ({combined_non_kw:.1%}) > keyword ({kw:.1%})")
        print(f"    System learned structural/semantic/transform policy")
    elif kw > combined_non_kw:
        print(f"  ✗ FAIL: Keyword posterior ({kw:.1%}) > non-keyword ({combined_non_kw:.1%})")
        if last_snapshot.get("best"):
            bt = last_snapshot["best"]["predicate_type"]
            bp = last_snapshot["best"]["posterior"]
            print(f"    Best candidate: {bt} (post={bp:.4f})")
            print(f"    System converged to keyword — dataset may not be hard enough")
    else:
        print(f"  ∼ UNCLEAR: Keyword ({kw:.1%}) = non-keyword ({combined_non_kw:.1%})")
        print(f"    System did not clearly favor any type")

    return {
        "experiment": exp_name,
        "keyword_posterior": kw,
        "non_keyword_posterior": combined_non_kw,
        "passed": combined_non_kw > kw,
    }


def main():
    parser = argparse.ArgumentParser(description="Analyze experiment results")
    parser.add_argument("path", nargs="?", help="Path to JSON result file")
    parser.add_argument("--all", action="store_true", help="Analyze all results in results/")
    args = parser.parse_args()

    if args.all:
        paths = sorted(RESULTS_DIR.glob("*_candidates.json"))
        if not paths:
            print(f"No result files found in {RESULTS_DIR}")
            return
        verdicts = []
        for p in paths:
            v = analyze(str(p))
            verdicts.append(v)
            print("\n" + "=" * 40)

        print(f"\n{'=' * 60}")
        print(f"  OVERALL SUMMARY ({len(verdicts)} experiments)")
        print(f"{'=' * 60}")
        for v in verdicts:
            status = "✓ PASS" if v["passed"] else "✗ FAIL"
            print(f"  {status} | {v['experiment']:<20} | kw={v['keyword_posterior']:.1%} non-kw={v['non_keyword_posterior']:.1%}")

    elif args.path:
        analyze(args.path)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
