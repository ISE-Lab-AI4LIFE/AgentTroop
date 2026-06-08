"""Shared prompt loading utility supporting RMCBench CSV format.

RMCBench CSV columns:
    pid,category,task,level description,level,prompt,malicious functionality,...

Auto-detects ``prompt`` column header (case-insensitive).
"""

from __future__ import annotations

import csv
import logging
import random
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)


def load_prompts(
    csv_path: str = "",
    n: Optional[int] = None,
    column: str = "",
    seed: int = 42,
) -> List[str]:
    """Load prompts from a CSV file.

    Args:
        csv_path: Path to CSV file.  Defaults to ``prompt.csv`` in project root.
        n: Number of prompts to return (``None`` = all).
        column: Column name.  If empty, auto-detect ("Goal" or "prompt").
        seed: Random seed for shuffling when *n* is set.

    Returns:
        A list of prompt strings (possibly shuffled + truncated).
    """
    path = _resolve_path(csv_path)
    col = column or _detect_column(path)
    prompts: List[str] = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            val = row.get(col, "").strip()
            if val:
                prompts.append(val)

    if n is not None and n < len(prompts):
        rng = random.Random(seed)
        rng.shuffle(prompts)
        prompts = prompts[:n]

    logger.info("Loaded %d prompts from %s (column=%s)", len(prompts), path, col)
    return prompts


def _resolve_path(csv_path: str) -> Path:
    if csv_path:
        p = Path(csv_path)
        if p.exists():
            return p
    root = Path(__file__).resolve().parent
    p = root / "prompt.csv"
    if p.exists():
        return p
    raise FileNotFoundError(
        f"No CSV found at {csv_path or 'prompt.csv'} "
        f"(resolved from {root})"
    )


def _detect_column(path: Path) -> str:
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        try:
            headers = next(reader)
        except StopIteration:
            return "prompt"
    headers = [h.strip() for h in headers]
    h_lower = [h.lower() for h in headers]
    if "prompt" in h_lower:
        return headers[h_lower.index("prompt")]
    return headers[0] if headers else "prompt"
