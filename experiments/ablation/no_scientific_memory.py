"""Ablation: no Scientific Memory transfer.

Disables theory storage and retrieval so that each campaign starts with
an empty Scientific Memory.  This isolates the contribution of cross-model
knowledge transfer to intervention efficiency.

Usage:
    python -m experiments.ablation.no_scientific_memory [--campaign no_sm_campaign]
"""

import argparse
import logging
from typing import Any, Dict, List, Optional

from knowledge.scientific_memory import ScientificMemory, Theory


class NoScientificMemory:
    """ScientificMemory variant that does not persist or retrieve theories.

    All store/read operations return empty results, mimicking a fresh
    Scientific Memory for every campaign.
    """

    def __init__(self, base_sm: Optional[ScientificMemory] = None) -> None:
        self._base = base_sm

    def save_theory(self, theory: Theory) -> str:
        return "noop"

    def get_theory(self, theory_id: str) -> Optional[Theory]:
        return None

    def find_theories(self, *args: Any, **kwargs: Any) -> List[Theory]:
        return []

    def find_theories_by_pattern(self, *args: Any, **kwargs: Any) -> List[Theory]:
        return []

    def get_all_theories(self, *args: Any, **kwargs: Any) -> List[Theory]:
        return []

    def compact_if_needed(self, *args: Any, **kwargs: Any) -> int:
        return 0

    def compact_older_than(self, *args: Any, **kwargs: Any) -> int:
        return 0

    def __getattr__(self, name: str) -> Any:
        if self._base is not None:
            return getattr(self._base, name)
        raise AttributeError(name)


def wrap_orchestrator_no_scientific_memory(orch: Any) -> Any:
    base_sm = orch.researcher.scientific_memory
    orch.researcher.scientific_memory = NoScientificMemory(base_sm)
    return orch


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="No Scientific Memory ablation")
    parser.add_argument("--campaign", default="ablation_no_scientific_memory", help="Campaign ID")
    parser.add_argument("--max-iter", type=int, default=20, help="Max iterations")
    args = parser.parse_args()
    print(f"Ablation: no_scientific_memory (campaign={args.campaign}, max_iter={args.max_iter})")
    print("Import wrap_orchestrator_no_scientific_memory() in your experiment harness.")
