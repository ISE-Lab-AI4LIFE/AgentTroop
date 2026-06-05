from typing import Dict, Set

from core.program import (
    AndNode,
    ApplyTransformNode,
    ClassifierNode,
    IfThenElseNode,
    Node,
    NotNode,
    OrNode,
    PredicateNode,
    Program,
    ThresholdNode,
)

from adapters.base_victim import BaseVictim


class PrimitiveDiscoveryEvaluator:
    """Evaluates how well a discovered program recovers the ground-truth primitives.
    
    Uses precision, recall, and F1 over primitive sets.
    """

    def primitive_set(self, program: Program) -> Set[str]:
        """Extract the set of (primitive_name, primitive_type) strings from a program."""
        primitives: Set[str] = set()

        def _walk(node: Node) -> None:
            if isinstance(node, PredicateNode):
                primitives.add(
                    f"predicate:{node.primitive.name}:{node.primitive.parameters}"
                )
            elif isinstance(node, ClassifierNode):
                primitives.add(f"classifier:{node.primitive.name}")
            elif isinstance(node, ThresholdNode):
                primitives.add(
                    f"threshold:{node.classifier.name}:{node.threshold}"
                )
            elif isinstance(node, ApplyTransformNode):
                primitives.add(f"transform:{node.transform.name}")
                _walk(node.inner)
            elif isinstance(node, AndNode):
                _walk(node.left)
                _walk(node.right)
            elif isinstance(node, OrNode):
                _walk(node.left)
                _walk(node.right)
            elif isinstance(node, NotNode):
                _walk(node.child)
            elif isinstance(node, IfThenElseNode):
                _walk(node.condition)

        _walk(program.root)
        return primitives

    def evaluate(
        self, victim: BaseVictim, discovered_program: Program
    ) -> Dict[str, float]:
        """Compute precision, recall, F1 for primitive discovery."""
        gt = victim.get_ground_truth_program()
        if gt is None:
            return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

        gt_set = self.primitive_set(gt)
        disc_set = self.primitive_set(discovered_program)

        true_positives = len(gt_set & disc_set)
        false_positives = len(disc_set - gt_set)
        false_negatives = len(gt_set - disc_set)

        precision = (
            true_positives / (true_positives + false_positives)
            if (true_positives + false_positives) > 0
            else 0.0
        )
        recall = (
            true_positives / (true_positives + false_negatives)
            if (true_positives + false_negatives) > 0
            else 0.0
        )
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        return {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "gt_primitive_count": len(gt_set),
            "discovered_primitive_count": len(disc_set),
        }
