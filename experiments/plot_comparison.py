#!/usr/bin/env python3
"""Plot RQ0 & RQ1 comparison across campaigns."""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

OUTPUTS = Path(__file__).resolve().parent.parent / "outputs" / "campaign"

CAMPAIGNS = [
    ("llama31_5seeds_20260614_173710",  "5 seeds\n(old config)"),
    ("llama31_50seeds_20260614_183321", "50 seeds\n(old config)"),
    ("llama31_50seeds_v2_20260614_230116", "50 seeds v2\n(new config)"),
    ("llama31_full_20260615_004702", "50 seeds full\n(new config)"),
]

def load_report(cid):
    p = OUTPUTS / cid / "evaluation" / "evaluation_report.json"
    if not p.exists():
        return None
    return json.loads(p.read_text())

def load_vs(cid):
    p = OUTPUTS / cid / "version_space.json"
    if not p.exists():
        return None
    return json.loads(p.read_text())

data = []
labels = []
for cid, label in CAMPAIGNS:
    r = load_report(cid)
    vs = load_vs(cid)
    if r is None:
        print(f"WARNING: no report for {cid}", file=sys.stderr)
        continue
    labels.append(label)
    data.append({
        "rq0": r["rq0"]["accuracy"],
        "rq0_passed": r["rq0"]["passed"],
        "rq1": r["rq1"]["best_accuracy"],
        "rq1_reached": r["rq1"]["reached"],
        "episodes": r["rq1"]["total_episodes"],
        "baseline_asr": r["asr"]["asr"],
        "harmony_asr": r["harmony_asr"]["asr"],
        "harmony_attempted": r["harmony_asr"]["attempted"],
        "harmony_total": r["harmony_asr"]["total"],
        "vs_entropy": vs.get("entropy", 0) if vs else 0,
        "vs_candidates": vs.get("num_candidates", 0) if vs else 0,
        "vs_updates": vs.get("num_updates", 0) if vs else 0,
    })

if len(data) < 2:
    print("Not enough campaigns with data", file=sys.stderr)
    sys.exit(1)

x = np.arange(len(data))
w = 0.25

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("Campaign Comparison: RQ0 & RQ1", fontsize=14, fontweight="bold")

# --- Left: RQ0 & RQ1 bar chart ---
b1 = ax1.bar(x - w/2, [d["rq0"] for d in data], w, label="RQ0 (accuracy)", color="#4C72B0")
b2 = ax1.bar(x + w/2, [d["rq1"] for d in data], w, label="RQ1 (best accuracy)", color="#DD8452")

for i, d in enumerate(data):
    for bar, val, label in [(b1[i], d["rq0"], "RQ0"), (b2[i], d["rq1"], "RQ1")]:
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                 f"{val:.2f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    # pass/fail markers
    passed = d["rq0_passed"]
    reached = d["rq1_reached"]
    ax1.text(x[i] - w/2, 0.02, "PASS" if passed else "FAIL",
             ha="center", va="bottom", fontsize=7,
             color="green" if passed else "red", fontweight="bold")
    ax1.text(x[i] + w/2, 0.02, "REACHED" if reached else "NOT",
             ha="center", va="bottom", fontsize=7,
             color="green" if reached else "red", fontweight="bold")

ax1.set_xticks(x)
ax1.set_xticklabels(labels)
ax1.set_ylabel("Score")
ax1.set_ylim(0, 1.15)
ax1.axhline(y=0.85, color="gray", linestyle="--", alpha=0.5, label="threshold=0.85")
ax1.legend(loc="upper left")
ax1.set_title("RQ0 (program accuracy) & RQ1 (intervention efficiency)")

# --- Right: Grouped multi-metric ---
metrics = ["baseline_asr", "harmony_asr", "vs_entropy", "vs_candidates", "episodes"]
metric_labels = ["Baseline ASR", "Harmony ASR", "VS Entropy", "VS Candidates", "Episodes"]
colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2"]

# normalize for display
vmax = [max(d.get(m, 0) for d in data) for m in metrics]
vmax = [v if v > 0 else 1 for v in vmax]

for i, (m, ml, c) in enumerate(zip(metrics, metric_labels, colors)):
    vals = [d.get(m, 0) / vmax[i] for d in data]
    actual = [d.get(m, 0) for d in data]
    offset = (i - len(metrics)/2 + 0.5) * (w / len(metrics))
    bars = ax2.bar(x + offset, vals, w/len(metrics), label=ml, color=c, alpha=0.85)
    for bar, a in zip(bars, actual):
        val_str = f"{a:.2f}" if isinstance(a, float) else str(a)
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                 val_str, ha="center", va="bottom", fontsize=6, rotation=45)

ax2.set_xticks(x)
ax2.set_xticklabels(labels)
ax2.set_ylabel("Normalized value")
ax2.legend(loc="upper left", fontsize=7)
ax2.set_title("Additional metrics (normalized)")

plt.tight_layout()
out_path = Path(__file__).resolve().parent / "plots"
out_path.mkdir(exist_ok=True)
dest = out_path / "campaign_comparison_rq0_rq1.png"
plt.savefig(dest, dpi=150, bbox_inches="tight")
print(f"Saved to {dest}")

# --- Also print a text table ---
print()
print(f"{'Campaign':<25} {'RQ0':>5} {'RQ1':>5} {'Ep':>5} {'bASR':>5} {'hASR':>5} {'Entropy':>7} {'Cand':>5}")
print("-" * 72)
for lbl, d in zip(labels, data):
    lbl_clean = lbl.replace("\n", " ")
    print(f"{lbl_clean:<25} {d['rq0']:>5.2f} {d['rq1']:>5.2f} {d['episodes']:>5} "
          f"{d['baseline_asr']:>5.2f} {d['harmony_asr']:>5.2f} "
          f"{d['vs_entropy']:>7.3f} {d['vs_candidates']:>5}")
