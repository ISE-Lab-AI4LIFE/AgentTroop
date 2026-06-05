from typing import List

from core.executor import ProgramExecutor
from core.primitive import default_registry
from core.program import Program


class ProgramEquivalenceChecker:
    """Compares two programs by their input-output behaviour."""

    def __init__(self) -> None:
        self._executor = ProgramExecutor(default_registry)

    def are_equivalent(
        self, p1: Program, p2: Program, test_inputs: List[str]
    ) -> bool:
        """Return True if both programs produce identical outcomes on all test inputs."""
        for inp in test_inputs:
            if self._executor.execute(p1, inp) != self._executor.execute(p2, inp):
                return False
        return True

    def equivalence_with_tolerance(
        self,
        p1: Program,
        p2: Program,
        test_inputs: List[str],
        tolerance: float = 0.01,
    ) -> float:
        """Return the fraction of test inputs where both programs agree."""
        if not test_inputs:
            return 1.0
        mismatches = 0
        for inp in test_inputs:
            if self._executor.execute(p1, inp) != self._executor.execute(p2, inp):
                mismatches += 1
        agreement = 1.0 - (mismatches / len(test_inputs))
        return agreement
