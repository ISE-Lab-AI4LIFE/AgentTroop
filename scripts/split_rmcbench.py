"""Split RMCBench prompt.csv into seed, deep-dive, and final test sets.

Usage:
    python scripts/split_rmcbench.py [--csv prompt.csv] [--seed-dir data/]
"""

import argparse
import csv
import logging
import os
import random
from pathlib import Path
from typing import List, Tuple

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def load_prompts(csv_path: str) -> List[Tuple[str, int]]:
    """Load prompts from RMCBench CSV. Returns list of (prompt, label).
    
    Label: 0 for benign, 1 for harmful (based on 'malicious' column or prompt content).
    """
    prompts: List[Tuple[str, int]] = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            prompt = (row.get("prompt") or row.get("Goal") or "").strip()
            if not prompt:
                continue
            level = row.get("level", "").strip().lower()
            malicious = row.get("malicious functionality", "").strip().lower()
            if "harmful" in level or "malicious" in malicious or "yes" in malicious:
                label = 1
            else:
                label = 0
            prompts.append((prompt, label))
    logger.info("Loaded %d prompts from %s", len(prompts), csv_path)
    return prompts


def split_prompts(
    prompts: List[Tuple[str, int]],
    seed_harmful: int = 5,
    seed_benign: int = 5,
    deep_harmful: int = 50,
    deep_benign: int = 50,
    test_harmful: int = 30,
    test_benign: int = 30,
    seed: int = 42,
) -> Tuple[List[Tuple[str, int]], List[Tuple[str, int]], List[Tuple[str, int]]]:
    """Split prompts into three non-overlapping sets."""
    rng = random.Random(seed)
    
    harmful = [(p, 1) for p, l in prompts if l == 1]
    benign = [(p, 0) for p, l in prompts if l == 0]
    
    rng.shuffle(harmful)
    rng.shuffle(benign)
    
    seed_set = harmful[:seed_harmful] + benign[:seed_benign]
    deep_set = harmful[seed_harmful:seed_harmful + deep_harmful] + benign[seed_benign:seed_benign + deep_benign]
    test_set = harmful[seed_harmful + deep_harmful:seed_harmful + deep_harmful + test_harmful] + benign[seed_benign + deep_benign:seed_benign + deep_benign + test_benign]
    
    rng.shuffle(seed_set)
    rng.shuffle(deep_set)
    rng.shuffle(test_set)
    
    logger.info(
        "Split: seed=%d (harmful=%d, benign=%d), deep=%d (harmful=%d, benign=%d), "
        "test=%d (harmful=%d, benign=%d)",
        len(seed_set), seed_harmful, seed_benign,
        len(deep_set), deep_harmful, deep_benign,
        len(test_set), test_harmful, test_benign,
    )
    
    return seed_set, deep_set, test_set


def save_csv(prompts: List[Tuple[str, int]], path: str) -> None:
    """Save prompts to CSV with 'prompt' and 'label' columns."""
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["prompt", "label"])
        for prompt, label in prompts:
            writer.writerow([prompt, label])
    logger.info("Saved %d prompts to %s", len(prompts), path)


def main():
    parser = argparse.ArgumentParser(description="Split RMCBench into seed/deep/test sets")
    parser.add_argument("--csv", default="prompt.csv", help="Path to RMCBench CSV")
    parser.add_argument("--seed-dir", default="data", help="Output directory")
    args = parser.parse_args()
    
    prompts = load_prompts(args.csv)
    
    os.makedirs(args.seed_dir, exist_ok=True)
    
    seed_set, deep_set, test_set = split_prompts(prompts)
    
    save_csv(seed_set, os.path.join(args.seed_dir, "seed_prompts.csv"))
    save_csv(deep_set, os.path.join(args.seed_dir, "deep_dive_prompts.csv"))
    save_csv(test_set, os.path.join(args.seed_dir, "final_test_prompts.csv"))
    
    logger.info("Split complete. Files saved to %s/", args.seed_dir)


if __name__ == "__main__":
    main()
