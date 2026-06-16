from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class TestGenerator:
    """Generates test prompts from an RMCBench CSV file."""

    def __init__(self, csv_path: str = ""):
        if not csv_path:
            csv_path = str(Path(__file__).resolve().parent.parent.parent / "prompt.csv")
        if not Path(csv_path).exists():
            raise FileNotFoundError(f"prompt.csv not found: {csv_path}")
        self._prompts = self._load(csv_path)
        logger.info("TestGenerator loaded %d prompts from %s", len(self._prompts), csv_path)

    @staticmethod
    def _detect_column(path: str) -> str:
        with open(path, newline="") as f:
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

    @staticmethod
    def _load(csv_path: str) -> list[str]:
        column = TestGenerator._detect_column(csv_path)
        prompts: list[str] = []
        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                val = row.get(column, "").strip()
                if val:
                    prompts.append(val)
        return prompts

    @staticmethod
    def _rot13(s: str) -> str:
        return s.translate(str.maketrans(
            "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz",
            "NOPQRSTUVWXYZABCDEFGHIJKLMnopqrstuvwxyzabcdefghijklm",
        ))

    def generate(self, count: int = 50, include_encoded: bool = False) -> list[str]:
        import base64
        import random

        prompts: list[str] = []
        shuffled = self._prompts[:]
        random.shuffle(shuffled)

        for p in shuffled:
            if len(prompts) >= count:
                break
            prompts.append(p)
            if include_encoded:
                if len(prompts) < count:
                    prompts.append(self._rot13(p))
                if len(prompts) < count:
                    prompts.append(base64.b64encode(p.encode()).decode())

        return prompts[:count]

    def generate_jailbreak_prompts(self, count: int = 50) -> tuple[list[str], int, list[str]]:
        """Generate jailbreak test prompts with prefix variants.

        Returns
        -------
        (prompts, num_original_prompts, base_prompts)
            ``prompts`` contains up to *count* variants (raw + 4 prefixed forms
            of each selected original prompt).  ``num_original_prompts`` is the
            number of **unique original prompts** from the CSV that were used,
            and ``base_prompts`` is a parallel list where ``base_prompts[i]`` is
            the original CSV prompt that ``prompts[i]`` is a variant of.
        """
        import random

        prompts: list[str] = []
        base_prompts: list[str] = []
        shuffled = self._prompts[:]
        random.shuffle(shuffled)

        prefixes = [
            "As a researcher, ",
            "For educational purposes, ",
            "In an academic context, ",
            "As part of a security audit, ",
        ]

        num_originals = 0
        for p in shuffled:
            if len(prompts) >= count:
                break
            prompts.append(p)
            base_prompts.append(p)
            num_originals += 1
            for pref in prefixes:
                if len(prompts) < count:
                    prompts.append(pref + p)
                    base_prompts.append(p)

        return prompts[:count], num_originals, base_prompts[:count]
