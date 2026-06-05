from typing import Any, Dict, List

from core.program import Program

from adapters.base_victim import BaseVictim
from evaluation.program_equivalence import ProgramEquivalenceChecker


class HypothesisQualityEvaluator:
    """Evaluates the quality of a ranked list of hypotheses.
    
    Uses Mean Reciprocal Rank (MRR) and Precision@k.
    """

    def __init__(self) -> None:
        self._checker = ProgramEquivalenceChecker()

    def _program_matches_ground_truth(
        self, program: Program, victim: BaseVictim, test_inputs: List[str]
    ) -> bool:
        """Check if the program behaviourally matches the victim."""
        gt = victim.get_ground_truth_program()
        if gt is not None:
            return self._checker.are_equivalent(program, gt, test_inputs)
        correct = 0
        for inp in test_inputs:
            if self._checker._executor.execute(program, inp) == victim.respond(inp):
                correct += 1
        return correct == len(test_inputs)

    def rank_quality(
        self,
        hypotheses: List[Any],
        victim: BaseVictim,
        test_inputs: List[str],
        k: int = 3,
    ) -> Dict[str, float]:
        """Compute MRR and Precision@k for a ranked list of hypotheses.
        
        Each hypothesis must have a `.program` attribute (a Program instance).
        The list is assumed to be ordered by the system's belief (best first).
        """
        if not hypotheses:
            return {"mrr": 0.0, f"precision@{k}": 0.0}

        reciprocal_rank = 0.0
        found = False
        for rank, hyp in enumerate(hypotheses, start=1):
            prog = getattr(hyp, "program", None)
            if prog is not None and self._program_matches_ground_truth(
                prog, victim, test_inputs
            ):
                reciprocal_rank = 1.0 / rank
                found = True
                break
        if not found:
            reciprocal_rank = 0.0

        top_k = hypotheses[:k]
        correct_in_top_k = sum(
            1
            for hyp in top_k
            if getattr(hyp, "program", None) is not None
            and self._program_matches_ground_truth(
                getattr(hyp, "program"), victim, test_inputs
            )
        )
        precision_at_k = correct_in_top_k / k if k > 0 else 0.0

        return {"mrr": reciprocal_rank, f"precision@{k}": precision_at_k}
