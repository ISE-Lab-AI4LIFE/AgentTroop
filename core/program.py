import abc
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Type

from .primitive import Classifier, Predicate, Primitive, Transform
from .types import Outcome, ProgramID


class Node(abc.ABC):
    @abc.abstractmethod
    def to_dict(self) -> Dict[str, Any]:
        raise NotImplementedError

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "Node":
        node_type = data["type"]
        mapping: Dict[str, Type[Node]] = {
            "PredicateNode": PredicateNode,
            "TransformNode": TransformNode,
            "ClassifierNode": ClassifierNode,
            "ThresholdNode": ThresholdNode,
            "ApplyTransformNode": ApplyTransformNode,
            "AndNode": AndNode,
            "OrNode": OrNode,
            "NotNode": NotNode,
            "IfThenElseNode": IfThenElseNode,
        }
        node_cls = mapping.get(node_type)
        if node_cls is None:
            raise ValueError(f"Unknown node type: {node_type}")
        return node_cls.from_dict(data)

    def __repr__(self) -> str:
        return str(self)


@dataclass
class AtomicNode(Node):
    primitive: Primitive

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": type(self).__name__,
            "primitive": self.primitive.to_dict(),
        }

    @staticmethod
    def primitive_from_dict(data: Dict[str, Any]) -> Primitive:
        primitive_name = data["name"]
        parameters = data.get("parameters", {})
        from .primitive import PrimitiveRegistry

        registry = PrimitiveRegistry()
        try:
            return registry.get(primitive_name, parameters)
        except ValueError:
            primitive_type = data.get("type")
            if primitive_type == "PredicateNode":
                return Predicate(name=primitive_name, parameters=parameters)
            if primitive_type == "TransformNode":
                return Transform(name=primitive_name, parameters=parameters)
            if primitive_type == "ClassifierNode":
                return Classifier(name=primitive_name, parameters=parameters)
            return Primitive(name=primitive_name, parameters=parameters)


@dataclass
class PredicateNode(AtomicNode):
    primitive: Predicate

    def __str__(self) -> str:
        return f"PredicateNode({self.primitive.name}, {self.primitive.parameters})"

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "PredicateNode":
        primitive_data = data["primitive"]
        primitive = AtomicNode.primitive_from_dict(primitive_data)
        return PredicateNode(primitive=primitive)


@dataclass
class TransformNode(AtomicNode):
    primitive: Transform

    def __str__(self) -> str:
        return f"TransformNode({self.primitive.name})"

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "TransformNode":
        primitive_data = data["primitive"]
        primitive = AtomicNode.primitive_from_dict(primitive_data)
        return TransformNode(primitive=primitive)


@dataclass
class ClassifierNode(AtomicNode):
    primitive: Classifier

    def __str__(self) -> str:
        return f"ClassifierNode({self.primitive.name})"

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "ClassifierNode":
        primitive_data = data["primitive"]
        primitive = AtomicNode.primitive_from_dict(primitive_data)
        return ClassifierNode(primitive=primitive)


@dataclass
class ThresholdNode(Node):
    classifier: Classifier
    threshold: float

    def __str__(self) -> str:
        return f"ThresholdNode({self.classifier.name} > {self.threshold})"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "ThresholdNode",
            "classifier": self.classifier.to_dict(),
            "threshold": self.threshold,
        }

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "ThresholdNode":
        primitive_data = data["classifier"]
        classifier = AtomicNode.primitive_from_dict(primitive_data)
        return ThresholdNode(classifier=classifier, threshold=float(data["threshold"]))


@dataclass
class ApplyTransformNode(Node):
    transform: Transform
    inner: Node

    def __str__(self) -> str:
        return f"ApplyTransformNode({self.transform.name}, {self.inner})"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "ApplyTransformNode",
            "transform": self.transform.to_dict(),
            "inner": self.inner.to_dict(),
        }

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "ApplyTransformNode":
        transform = AtomicNode.primitive_from_dict(data["transform"])
        inner = Node.from_dict(data["inner"])
        return ApplyTransformNode(transform=transform, inner=inner)


@dataclass
class BinaryNode(Node):
    left: Node
    right: Node

    def children(self) -> List[Node]:
        return [self.left, self.right]

    def sort_children(self) -> None:
        ordered = sorted(self.children(), key=lambda node: str(node))
        self.left, self.right = ordered[0], ordered[1]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": type(self).__name__,
            "left": self.left.to_dict(),
            "right": self.right.to_dict(),
        }

    @staticmethod
    def from_dict(data: Dict[str, Any], cls: Type["BinaryNode"]) -> "BinaryNode":
        left = Node.from_dict(data["left"])
        right = Node.from_dict(data["right"])
        return cls(left=left, right=right)


@dataclass
class AndNode(BinaryNode):
    def __str__(self) -> str:
        return f"AndNode({self.left}, {self.right})"

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "AndNode":
        return BinaryNode.from_dict(data, AndNode)  # type: ignore[return-value]


@dataclass
class OrNode(BinaryNode):
    def __str__(self) -> str:
        return f"OrNode({self.left}, {self.right})"

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "OrNode":
        return BinaryNode.from_dict(data, OrNode)  # type: ignore[return-value]


@dataclass
class NotNode(Node):
    child: Node

    def __str__(self) -> str:
        return f"NotNode({self.child})"

    def to_dict(self) -> Dict[str, Any]:
        return {"type": "NotNode", "child": self.child.to_dict()}

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "NotNode":
        child = Node.from_dict(data["child"])
        return NotNode(child=child)


@dataclass
class IfThenElseNode(Node):
    condition: Node
    then_outcome: Outcome = 1
    else_outcome: Outcome = 0

    def __str__(self) -> str:
        return f"IfThenElseNode(IF {self.condition} THEN {self.then_outcome} ELSE {self.else_outcome})"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "IfThenElseNode",
            "condition": self.condition.to_dict(),
            "then_outcome": self.then_outcome,
            "else_outcome": self.else_outcome,
        }

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "IfThenElseNode":
        condition = Node.from_dict(data["condition"])
        return IfThenElseNode(
            condition=condition,
            then_outcome=int(data.get("then_outcome", 1)),
            else_outcome=int(data.get("else_outcome", 0)),
        )


@dataclass
class Program:
    root: IfThenElseNode
    id: ProgramID = field(default_factory=lambda: f"prog_{int(time.time() * 1000)}")
    version_id: str = "1.0"
    created_at: float = field(default_factory=time.time)
    deprecated_at: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "version_id": self.version_id,
            "created_at": self.created_at,
            "deprecated_at": self.deprecated_at,
            "metadata": self.metadata,
            "root": self.root.to_dict(),
        }

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "Program":
        root = IfThenElseNode.from_dict(data["root"])
        return Program(
            root=root,
            id=data.get("id", f"prog_{int(time.time() * 1000)}"),
            version_id=data.get("version_id", "1.0"),
            created_at=float(data.get("created_at", time.time())),
            deprecated_at=data.get("deprecated_at"),
            metadata=data.get("metadata", {}),
        )

    def canonicalize(self) -> "Program":
        self.root.condition = self._canonicalize_node(self.root.condition)
        return self

    def _canonicalize_node(self, node: Node) -> Node:
        if isinstance(node, BinaryNode):
            node.left = self._canonicalize_node(node.left)
            node.right = self._canonicalize_node(node.right)
            node.sort_children()
            return node
        if isinstance(node, NotNode):
            node.child = self._canonicalize_node(node.child)
            if isinstance(node.child, NotNode):
                return self._canonicalize_node(node.child.child)
            return node
        if isinstance(node, ApplyTransformNode):
            node.inner = self._canonicalize_node(node.inner)
            return node
        if isinstance(node, IfThenElseNode):
            node.condition = self._canonicalize_node(node.condition)
            return node
        if isinstance(node, ThresholdNode):
            node.threshold = round(node.threshold, 6)
            return node
        return node

    def complexity(self) -> int:
        return self._count_nodes(self.root)

    def depth(self) -> int:
        return self._measure_depth(self.root)

    def primitive_count(self) -> int:
        return self._count_primitives(self.root)

    def mdl_score(self, alpha: float = 0.1) -> float:
        return self.complexity() + alpha * len(repr(self))

    def complexity_metrics(self) -> Dict[str, Any]:
        return {
            "node_count": self.complexity(),
            "depth": self.depth(),
            "primitive_count": self.primitive_count(),
            "mdl_score": self.mdl_score(),
        }

    def _count_nodes(self, node: Node) -> int:
        if isinstance(node, AtomicNode):
            return 1
        if isinstance(node, ThresholdNode):
            return 1
        if isinstance(node, ApplyTransformNode):
            return 1 + self._count_nodes(node.inner)
        if isinstance(node, BinaryNode):
            return 1 + self._count_nodes(node.left) + self._count_nodes(node.right)
        if isinstance(node, NotNode):
            return 1 + self._count_nodes(node.child)
        if isinstance(node, IfThenElseNode):
            return 1 + self._count_nodes(node.condition)
        return 1

    def _measure_depth(self, node: Node) -> int:
        if isinstance(node, AtomicNode) or isinstance(node, ThresholdNode):
            return 1
        if isinstance(node, ApplyTransformNode):
            return 1 + self._measure_depth(node.inner)
        if isinstance(node, BinaryNode):
            return 1 + max(self._measure_depth(node.left), self._measure_depth(node.right))
        if isinstance(node, NotNode):
            return 1 + self._measure_depth(node.child)
        if isinstance(node, IfThenElseNode):
            return 1 + self._measure_depth(node.condition)
        return 1

    def _count_primitives(self, node: Node) -> int:
        if isinstance(node, AtomicNode):
            return 1
        if isinstance(node, ThresholdNode):
            return 1
        if isinstance(node, ApplyTransformNode):
            return 1 + self._count_primitives(node.inner)
        if isinstance(node, BinaryNode):
            return self._count_primitives(node.left) + self._count_primitives(node.right)
        if isinstance(node, NotNode):
            return self._count_primitives(node.child)
        if isinstance(node, IfThenElseNode):
            return self._count_primitives(node.condition)
        return 0

    def canonical_form(self) -> str:
        self.canonicalize()
        return repr(self)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Program):
            return False
        return self.canonical_form() == other.canonical_form()

    def __hash__(self) -> int:
        return hash(self.canonical_form())

    def __repr__(self) -> str:
        return f"Program(id={self.id}, root={self.root})"


@dataclass
class ProgramFragment:
    id: str
    root: Node
    description: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "description": self.description,
            "metadata": self.metadata,
            "root": self.root.to_dict(),
        }

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "ProgramFragment":
        return ProgramFragment(
            id=data["id"],
            root=Node.from_dict(data["root"]),
            description=data.get("description", ""),
            metadata=data.get("metadata", {}),
        )


@dataclass
class PolicyTemplate:
    id: str
    name: str
    description: str
    parameter_names: List[str]
    base_program: Program
    version_id: str = "1.0"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def instantiate(self, **parameters: Any) -> Program:
        return self.base_program

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "parameter_names": self.parameter_names,
            "version_id": self.version_id,
            "metadata": self.metadata,
            "base_program": self.base_program.to_dict(),
        }

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "PolicyTemplate":
        return PolicyTemplate(
            id=data["id"],
            name=data["name"],
            description=data["description"],
            parameter_names=list(data.get("parameter_names", [])),
            base_program=Program.from_dict(data["base_program"]),
            version_id=data.get("version_id", "1.0"),
            metadata=data.get("metadata", {}),
        )


@dataclass
class Policy:
    id: str
    name: str
    description: str
    program: Program
    version_id: str = "1.0"
    created_at: float = field(default_factory=time.time)
    deprecated_at: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "program": self.program.to_dict(),
            "version_id": self.version_id,
            "created_at": self.created_at,
            "deprecated_at": self.deprecated_at,
            "metadata": self.metadata,
        }

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "Policy":
        return Policy(
            id=data["id"],
            name=data["name"],
            description=data["description"],
            program=Program.from_dict(data["program"]),
            version_id=data.get("version_id", "1.0"),
            created_at=float(data.get("created_at", time.time())),
            deprecated_at=data.get("deprecated_at"),
            metadata=data.get("metadata", {}),
        )
