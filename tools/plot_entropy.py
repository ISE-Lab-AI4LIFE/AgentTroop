#!/usr/bin/env python3
"""Plot and compare entropy trajectories from two CSV files.

Usage:
    python3 tools/plot_entropy.py entropy.csv entropy_1.csv
    python3 tools/plot_entropy.py entropy.csv entropy_1.csv -o compare.png
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def load(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if "index" not in df.columns or "H_after" not in df.columns:
        print(f"Error: {csv_path} must have 'index' and 'H_after' columns", file=sys.stderr)
        sys.exit(1)
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare entropy trajectories")
    parser.add_argument("csv_a", help="First entropy CSV")
    parser.add_argument("csv_b", help="Second entropy CSV")
    parser.add_argument("-o", "--output", default="entropy_comparison.png",
                        help="Output PNG path")
    parser.add_argument("--label-a", default=None, help="Label for first CSV")
    parser.add_argument("--label-b", default=None, help="Label for second CSV")
    parser.add_argument("--dpi", type=int, default=150)
    args = parser.parse_args()

    df_a = load(args.csv_a)
    df_b = load(args.csv_b)

    label_a = args.label_a or Path(args.csv_a).stem
    label_b = args.label_b or Path(args.csv_b).stem

    fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

    # ── Top: H_after trajectory ──
    ax = axes[0]
    ax.plot(df_a["index"], df_a["H_after"], marker="o", linestyle="-",
            label=label_a, alpha=0.85)
    ax.plot(df_b["index"], df_b["H_after"], marker="s", linestyle="-",
            label=label_b, alpha=0.85)
    ax.set_ylabel("H (entropy)")
    ax.set_title("Entropy (H_after) Trajectory Comparison")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # ── Bottom: IG comparison ──
    ax = axes[1]
    ax.bar(df_a["index"], df_a["IG"], width=0.35, alpha=0.7, label=label_a,
           align="center")
    ax.bar(df_b["index"], df_b["IG"], width=0.35, alpha=0.7, label=label_b,
           align="edge")
    ax.axhline(0, color="gray", linewidth=0.8)
    ax.set_xlabel("Belief update index")
    ax.set_ylabel("Information Gain (IG)")
    ax.set_title("Information Gain per Update")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # ── Summary stats ──
    print(f"{label_a}: {len(df_a)} updates, "
          f"H_start={df_a['H_before'].iloc[0]:.4f}, "
          f"H_end={df_a['H_after'].iloc[-1]:.4f}, "
          f"mean_IG={df_a['IG'].mean():+.4f}")
    print(f"{label_b}: {len(df_b)} updates, "
          f"H_start={df_b['H_before'].iloc[0]:.4f}, "
          f"H_end={df_b['H_after'].iloc[-1]:.4f}, "
          f"mean_IG={df_b['IG'].mean():+.4f}")

    plt.tight_layout()
    fig.savefig(args.output, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
