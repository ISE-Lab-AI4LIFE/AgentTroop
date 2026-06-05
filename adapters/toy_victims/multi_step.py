from typing import List

from core.executor import ProgramExecutor
from core.primitive import (
    Predicate,
    RemovePunctuationTransform,
    ToLowercaseTransform,
    Transform,
    default_registry,
)
from core.program import (
    ApplyTransformNode,
    IfThenElseNode,
    Node,
    PredicateNode,
    Program,
)
from core.types import Outcome

from adapters.base_victim import BaseVictim


class DecodeThenFilterVictim(BaseVictim):
    """Applies a sequence of transforms then checks a predicate.
    
    Simulates a safety pipeline that first decodes/transforms the prompt
    before applying a keyword or pattern filter.
    """

    def __init__(self, transforms: List[Transform], predicate: Predicate) -> None:
        super().__init__()
        self.transforms = transforms
        self.predicate = predicate
        inner: Node = PredicateNode(primitive=predicate)
        for t in reversed(transforms):
            inner = ApplyTransformNode(transform=t, inner=inner)
        self._program = Program(
            root=IfThenElseNode(condition=inner, then_outcome=1, else_outcome=0)
        )

    def respond(self, prompt: str) -> Outcome:
        return self._executor.execute(self._program, prompt)

    def get_metadata(self) -> dict:
        return {
            **super().get_metadata(),
            "type": "multi_step",
            "pipeline": "decode_then_filter",
            "num_transforms": len(self.transforms),
            "transform_names": [t.name for t in self.transforms],
            "predicate_name": self.predicate.name,
        }


class NormalizeThenFilterVictim(BaseVictim):
    """Normalizes a prompt (lowercase, remove punctuation) then applies a predicate.
    
    Simulates a common pre-processing pipeline before safety filtering.
    """

    def __init__(self, predicate: Predicate) -> None:
        super().__init__()
        self.predicate = predicate
        inner: Node = PredicateNode(primitive=predicate)
        inner = ApplyTransformNode(
            transform=RemovePunctuationTransform(), inner=inner
        )
        inner = ApplyTransformNode(
            transform=ToLowercaseTransform(), inner=inner
        )
        self._program = Program(
            root=IfThenElseNode(condition=inner, then_outcome=1, else_outcome=0)
        )

    def respond(self, prompt: str) -> Outcome:
        return self._executor.execute(self._program, prompt)

    def get_metadata(self) -> dict:
        return {
            **super().get_metadata(),
            "type": "multi_step",
            "pipeline": "normalize_then_filter",
            "predicate_name": self.predicate.name,
        }
