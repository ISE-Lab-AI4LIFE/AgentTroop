from typing import Any, Dict, List, Tuple

from .primitive import Classifier, Predicate, PrimitiveRegistry, Transform
from .program import ApplyTransformNode, AndNode, ClassifierNode, IfThenElseNode, NotNode, OrNode, PredicateNode, Program, ThresholdNode
from .types import Outcome


class ProgramExecutor:
    def __init__(self, registry: PrimitiveRegistry) -> None:
        self.registry = registry

    def execute(self, program: Program, prompt: str) -> Outcome:
        outcome, _ = self.execute_with_trace(program, prompt)
        return outcome

    def execute_with_trace(self, program: Program, prompt: str) -> Tuple[Outcome, List[Dict[str, Any]]]:
        trace: List[Dict[str, Any]] = []
        result = self._evaluate_node(program.root, prompt, trace)
        return int(result), trace

    def _evaluate_node(self, node: Any, prompt: str, trace: List[Dict[str, Any]]) -> Any:
        if isinstance(node, IfThenElseNode):
            condition_value = self._evaluate_node(node.condition, prompt, trace)
            trace.append({"node": repr(node), "condition": condition_value})
            return node.then_outcome if condition_value else node.else_outcome

        if isinstance(node, PredicateNode):
            value = node.primitive.evaluate(prompt)
            trace.append({"node": repr(node), "value": value})
            return value

        if isinstance(node, ClassifierNode):
            value = node.primitive.evaluate(prompt)
            trace.append({"node": repr(node), "score": value})
            return value

        if isinstance(node, ThresholdNode):
            score = node.classifier.evaluate(prompt)
            value = score > node.threshold
            trace.append({"node": repr(node), "score": score, "threshold": node.threshold, "value": value})
            return value

        if isinstance(node, ApplyTransformNode):
            transformed = node.transform.evaluate(prompt)
            trace.append({"node": repr(node), "transformed_prompt": transformed})
            return self._evaluate_node(node.inner, transformed, trace)

        if isinstance(node, AndNode):
            left_value = self._evaluate_node(node.left, prompt, trace)
            right_value = self._evaluate_node(node.right, prompt, trace)
            result = bool(left_value) and bool(right_value)
            trace.append({"node": repr(node), "left": left_value, "right": right_value, "value": result})
            return result

        if isinstance(node, OrNode):
            left_value = self._evaluate_node(node.left, prompt, trace)
            right_value = self._evaluate_node(node.right, prompt, trace)
            result = bool(left_value) or bool(right_value)
            trace.append({"node": repr(node), "left": left_value, "right": right_value, "value": result})
            return result

        if isinstance(node, NotNode):
            child_value = self._evaluate_node(node.child, prompt, trace)
            result = not bool(child_value)
            trace.append({"node": repr(node), "child": child_value, "value": result})
            return result

        raise ValueError(f"Unsupported node type: {type(node).__name__}")
