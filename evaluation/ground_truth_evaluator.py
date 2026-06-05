import difflib
from typing import List, Optional

from core.executor import ProgramExecutor
from core.primitive import default_registry
from core.program import Program

from adapters.base_victim import BaseVictim


class GroundTruthEvaluator:
    """Evaluates a discovered program against a victim's ground truth.
    
    Two modes of comparison:
    - behavioural: accuracy on test prompts
    - structural: canonical-form similarity & tree edit distance approximation
    """

    def __init__(
        self, victim: BaseVictim, discovered_program: Program
    ) -> None:
        self.victim = victim
        self.discovered = discovered_program
        self._executor = ProgramExecutor(default_registry)

    def compute_accuracy(self, test_prompts: List[str]) -> float:
        """Fraction of test prompts where discovered_program matches victim."""
        if not test_prompts:
            return 0.0
        correct = 0
        for prompt in test_prompts:
            expected = self.victim.respond(prompt)
            actual = self._executor.execute(self.discovered, prompt)
            if expected == actual:
                correct += 1
        return correct / len(test_prompts)

    def compute_program_similarity(self) -> float:
        """Compare the discovered program to the ground truth program.
        
        Uses canonical form string similarity when ground truth is available.
        Returns 0.0 if no ground truth program exists (e.g. neural victim).
        """
        gt = self.victim.get_ground_truth_program()
        if gt is None:
            return 0.0
        discovered_canon = self.discovered.canonical_form()
        gt_canon = gt.canonical_form()
        similarity = difflib.SequenceMatcher(
            None, discovered_canon, gt_canon
        ).ratio()
        return similarity
