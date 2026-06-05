from typing import List

from core.executor import ProgramExecutor
from core.primitive import Classifier, default_registry
from core.program import (
    AndNode,
    IfThenElseNode,
    NotNode,
    OrNode,
    Program,
    ThresholdNode,
)
from core.types import Outcome

from adapters.base_victim import BaseVictim


class AndVictim(BaseVictim):
    """Refuses a prompt if ALL sub-victims refuse it.
    
    Composes ground truth programs using AND logic.
    """

    def __init__(self, victims: List[BaseVictim]) -> None:
        super().__init__()
        self.victims = victims
        conditions = []
        for v in victims:
            gt = v.get_ground_truth_program()
            if gt is None:
                raise ValueError(
                    f"{type(v).__name__} has no ground truth program"
                )
            conditions.append(gt.root.condition)
        condition = conditions[0]
        for c in conditions[1:]:
            condition = AndNode(left=condition, right=c)
        self._program = Program(
            root=IfThenElseNode(condition=condition, then_outcome=1, else_outcome=0)
        )

    def respond(self, prompt: str) -> Outcome:
        return self._executor.execute(self._program, prompt)

    def get_metadata(self) -> dict:
        return {
            **super().get_metadata(),
            "type": "hybrid",
            "logic": "AND",
            "num_children": len(self.victims),
        }


class OrVictim(BaseVictim):
    """Refuses a prompt if ANY sub-victim refuses it.
    
    Composes ground truth programs using OR logic.
    """

    def __init__(self, victims: List[BaseVictim]) -> None:
        super().__init__()
        self.victims = victims
        conditions = []
        for v in victims:
            gt = v.get_ground_truth_program()
            if gt is None:
                raise ValueError(
                    f"{type(v).__name__} has no ground truth program"
                )
            conditions.append(gt.root.condition)
        condition = conditions[0]
        for c in conditions[1:]:
            condition = OrNode(left=condition, right=c)
        self._program = Program(
            root=IfThenElseNode(condition=condition, then_outcome=1, else_outcome=0)
        )

    def respond(self, prompt: str) -> Outcome:
        return self._executor.execute(self._program, prompt)

    def get_metadata(self) -> dict:
        return {
            **super().get_metadata(),
            "type": "hybrid",
            "logic": "OR",
            "num_children": len(self.victims),
        }


class NotVictim(BaseVictim):
    """Inverts the decision of a sub-victim.
    
    ACCEPT becomes REFUSE and vice versa.
    """

    def __init__(self, victim: BaseVictim) -> None:
        super().__init__()
        self.victim = victim
        gt = victim.get_ground_truth_program()
        if gt is None:
            raise ValueError(
                f"{type(victim).__name__} has no ground truth program"
            )
        condition = NotNode(child=gt.root.condition)
        self._program = Program(
            root=IfThenElseNode(condition=condition, then_outcome=1, else_outcome=0)
        )

    def respond(self, prompt: str) -> Outcome:
        return self._executor.execute(self._program, prompt)

    def get_metadata(self) -> dict:
        return {
            **super().get_metadata(),
            "type": "hybrid",
            "logic": "NOT",
        }


class ThresholdVictim(BaseVictim):
    """Refuses a prompt if a classifier score exceeds a threshold.
    
    Uses ThresholdNode in the ground truth program.
    """

    def __init__(self, classifier: Classifier, threshold: float) -> None:
        super().__init__()
        self.classifier = classifier
        self.threshold = threshold
        condition = ThresholdNode(classifier=classifier, threshold=threshold)
        self._program = Program(
            root=IfThenElseNode(condition=condition, then_outcome=1, else_outcome=0)
        )

    def respond(self, prompt: str) -> Outcome:
        return self._executor.execute(self._program, prompt)

    def get_metadata(self) -> dict:
        return {
            **super().get_metadata(),
            "type": "hybrid",
            "logic": "threshold",
            "classifier_name": self.classifier.name,
            "threshold": self.threshold,
        }
