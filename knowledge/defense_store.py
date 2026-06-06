from __future__ import annotations

import json
import logging
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Type

from neo4j import GraphDatabase, Session
from neo4j.exceptions import ServiceUnavailable

from core.program import (
    AndNode,
    ApplyTransformNode,
    AtomicNode,
    BinaryNode,
    ClassifierNode,
    IfThenElseNode,
    Node,
    NotNode,
    OrNode,
    PredicateNode,
    Program,
    ThresholdNode,
    TransformNode,
)

logger = logging.getLogger(__name__)


@dataclass
class DefenseProgramRecord:
    name: str
    program: Program
    confidence: float = 0.0
    provenance: List[str] = field(default_factory=list)
    id: str = ""
    version: int = 1
    created_at: float = field(default_factory=lambda: __import__("time").time())
    updated_at: float = field(default_factory=lambda: __import__("time").time())
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.id:
            self.id = f"dfp_{uuid.uuid4().hex[:12]}"
        self.confidence = max(0.0, min(1.0, float(self.confidence)))
        self.version = max(1, int(self.version))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "version": self.version,
            "confidence": self.confidence,
            "provenance": self.provenance,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "metadata": self.metadata,
            "program": self.program.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DefenseProgramRecord":
        return cls(
            id=data.get("id", ""),
            name=data.get("name", ""),
            version=int(data.get("version", 1)),
            confidence=float(data.get("confidence", 0.0)),
            provenance=list(data.get("provenance", [])),
            created_at=float(data.get("created_at", 0.0)),
            updated_at=float(data.get("updated_at", 0.0)),
            metadata=dict(data.get("metadata", {})),
            program=Program.from_dict(data["program"]),
        )


class DefenseProgramStore:
    def __init__(
        self,
        uri: str = "bolt://localhost:7687",
        user: str = "neo4j",
        password: str = "password",
        database: str = "neo4j",
    ) -> None:
        self.uri = uri
        self.user = user
        self.password = password
        self.database = database
        self._driver: Optional[Any] = None

    def _get_driver(self) -> Any:
        if self._driver is None:
            try:
                self._driver = GraphDatabase.driver(
                    self.uri, auth=(self.user, self.password)
                )
            except ServiceUnavailable as exc:
                raise ConnectionError(
                    f"Cannot connect to Neo4j at {self.uri}: {exc}"
                ) from exc
        return self._driver

    def _session(self) -> Session:
        driver = self._get_driver()
        return driver.session(database=self.database)

    def close(self) -> None:
        if self._driver is not None:
            self._driver.close()
            self._driver = None

    def __enter__(self) -> "DefenseProgramStore":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    def _ensure_constraints(self) -> None:
        with self._session() as session:
            result = session.run(
                "SHOW CONSTRAINTS WHERE type = 'NODE_PROPERTY_UNIQUENESS' "
                "AND labelsOrTypes = ['DefenseProgram'] "
                "AND properties = ['id']"
            )
            for record in result:
                name = record.get("name", "")
                if name:
                    session.run(f"DROP CONSTRAINT `{name}`")

            session.run(
                "CREATE CONSTRAINT IF NOT EXISTS "
                "FOR (p:DefenseProgram) REQUIRE (p.id, p.version) IS UNIQUE"
            )
            session.run(
                "CREATE CONSTRAINT IF NOT EXISTS "
                "FOR (n:ASTNode) REQUIRE n.node_id IS UNIQUE"
            )
            session.run(
                "CREATE CONSTRAINT IF NOT EXISTS "
                "FOR (p:PrimitiveNode) REQUIRE p.name IS UNIQUE"
            )
            session.run(
                "CREATE INDEX IF NOT EXISTS FOR (p:DefenseProgram) "
                "ON (p.confidence)"
            )
            session.run(
                "CREATE INDEX IF NOT EXISTS FOR (p:DefenseProgram) "
                "ON (p.name)"
            )

    def save(
        self,
        record: DefenseProgramRecord,
    ) -> str:
        record.confidence = max(0.0, min(1.0, float(record.confidence)))
        record.version = max(1, int(record.version))
        now = __import__("time").time()
        record.updated_at = now

        existing = self._get_latest_node(record.id)
        if existing is not None:
            record.version = existing.get("version", 0) + 1
            record.created_at = float(existing.get("created_at", now))
        else:
            record.version = 1
            record.created_at = now
            record.updated_at = now

        with self._session() as session:
            tx = session.begin_transaction()
            try:
                tx.run(
                    """CREATE (p:DefenseProgram {
                        id: $id,
                        name: $name,
                        version: $version,
                        confidence: $confidence,
                        provenance: $provenance,
                        created_at: $created_at,
                        updated_at: $updated_at,
                        metadata: $metadata
                    })""",
                    id=record.id,
                    name=record.name,
                    version=record.version,
                    confidence=record.confidence,
                    provenance=json.dumps(record.provenance, ensure_ascii=False),
                    created_at=record.created_at,
                    updated_at=record.updated_at,
                    metadata=json.dumps(record.metadata, ensure_ascii=False),
                )

                root_node_id = self._ast_to_graph(tx, record.program.root)

                tx.run(
                    "MATCH (p:DefenseProgram {id: $pid, version: $ver}), "
                    "(r:ASTNode {node_id: $rid}) "
                    "CREATE (p)-[:HAS_ROOT]->(r)",
                    pid=record.id, ver=record.version, rid=root_node_id,
                )

                if existing is not None:
                    prev_id = existing.element_id
                    tx.run(
                        """MATCH (prev) WHERE elementId(prev) = $prev_id
                           MATCH (cur:DefenseProgram {id: $id, version: $ver})
                           MERGE (prev)-[:NEXT_VERSION]->(cur)""",
                        prev_id=prev_id,
                        id=record.id,
                        ver=record.version,
                    )

                tx.commit()
            except Exception:
                tx.rollback()
                raise

        logger.debug("Saved defense program %s version %d", record.id, record.version)
        return record.id

    def get(
        self, program_id: str, version: Optional[int] = None
    ) -> Optional[DefenseProgramRecord]:
        if version is not None:
            node = self._get_version_node(program_id, version)
        else:
            node = self._get_latest_node(program_id)

        if node is None:
            return None
        return self._node_to_record(node)

    def delete(self, program_id: str) -> bool:
        with self._session() as session:
            result = session.run(
                "MATCH (p:DefenseProgram {id: $id}) "
                "OPTIONAL MATCH (p)-[:HAS_ROOT]->(root:ASTNode) "
                "OPTIONAL MATCH (all_ast:ASTNode) "
                "  WHERE all_ast.node_id STARTS WITH $prefix "
                "DETACH DELETE p, all_ast "
                "RETURN count(DISTINCT p) AS deleted",
                id=program_id,
                prefix=f"ast_{program_id}_",
            )
            record = result.single()
            return record is not None and record["deleted"] > 0

    def delete_all(self) -> int:
        with self._session() as session:
            result = session.run(
                "MATCH (p:DefenseProgram) DETACH DELETE p "
                "RETURN count(p) AS deleted"
            )
            record = result.single()
            deleted = record["deleted"] if record else 0

            session.run("MATCH (n:ASTNode) DETACH DELETE n")
            session.run("MATCH (p:PrimitiveNode) DETACH DELETE p")
            return deleted

    def find_by_confidence(
        self, min_confidence: float = 0.0
    ) -> List[DefenseProgramRecord]:
        with self._session() as session:
            result = session.run(
                """MATCH (p:DefenseProgram)
                   WHERE p.confidence >= $min_conf
                   AND NOT EXISTS ((p)-[:NEXT_VERSION]->())
                   RETURN p
                   ORDER BY p.confidence DESC, p.updated_at DESC""",
                min_conf=min_confidence,
            )
            return [self._node_to_record(rec["p"]) for rec in result]

    def find_by_primitive(
        self, primitive_name: str, min_confidence: float = 0.0
    ) -> List[DefenseProgramRecord]:
        with self._session() as session:
            result = session.run(
                """MATCH (p:DefenseProgram)
                   WHERE NOT EXISTS ((p)-[:NEXT_VERSION]->())
                   AND p.confidence >= $min_conf
                   AND EXISTS {
                       MATCH (p)-[:HAS_ROOT]->(root:ASTNode)
                       MATCH (root)-[:LEFT|RIGHT|CHILD|CONDITION|INNER*0..]->(n:ASTNode)
                       MATCH (n)-[:PRIMITIVE]->(pn:PrimitiveNode {name: $pn_name})
                   }
                   RETURN p
                   ORDER BY p.confidence DESC""",
                pn_name=primitive_name,
                min_conf=min_confidence,
            )
            return [self._node_to_record(rec["p"]) for rec in result]

    def list_program_ids(self) -> List[str]:
        with self._session() as session:
            result = session.run(
                """MATCH (p:DefenseProgram)
                   WHERE NOT EXISTS ((p)-[:NEXT_VERSION]->())
                   RETURN p.id AS id ORDER BY p.name"""
            )
            return [rec["id"] for rec in result]

    def update_confidence(self, program_id: str, new_confidence: float) -> bool:
        existing = self._get_latest_node(program_id)
        if existing is None:
            return False
        record = self._node_to_record(existing)
        record.confidence = new_confidence
        self.save(record)
        return True

    def export_programs(
        self, file_path: str, include_history: bool = True
    ) -> None:
        if include_history:
            query = (
                "MATCH (p:DefenseProgram) RETURN p "
                "ORDER BY p.id, p.version"
            )
        else:
            query = (
                "MATCH (p:DefenseProgram) "
                "WHERE NOT EXISTS ((p)-[:NEXT_VERSION]->()) "
                "RETURN p ORDER BY p.updated_at DESC"
            )

        with self._session() as session:
            result = session.run(query)
            records = [
                self._node_to_record(rec["p"]).to_dict()
                for rec in result
            ]

        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(records, f, indent=2, ensure_ascii=False)
        logger.info("Exported %d programs to %s", len(records), file_path)

    def import_programs(
        self, file_path: str, overwrite_existing: bool = False
    ) -> int:
        with open(file_path, "r", encoding="utf-8") as f:
            raw_list = json.load(f)

        count = 0
        for raw in raw_list:
            pid = raw.get("id", "")
            ver = int(raw.get("version", 1))
            if not overwrite_existing:
                existing = self.get(pid, version=ver)
                if existing is not None:
                    continue
            record = DefenseProgramRecord.from_dict(raw)
            self.save(record)
            count += 1
        logger.info("Imported %d programs from %s", count, file_path)
        return count

    def _get_latest_node(self, program_id: str) -> Any:
        with self._session() as session:
            result = session.run(
                """MATCH (p:DefenseProgram {id: $id})
                   WHERE NOT EXISTS ((p)-[:NEXT_VERSION]->())
                   RETURN p""",
                id=program_id,
            )
            record = result.single()
            return record["p"] if record else None

    def _get_version_node(self, program_id: str, version: int) -> Any:
        with self._session() as session:
            result = session.run(
                "MATCH (p:DefenseProgram {id: $id, version: $ver}) RETURN p",
                id=program_id,
                ver=version,
            )
            record = result.single()
            return record["p"] if record else None

    def _node_to_record(self, node: Any) -> DefenseProgramRecord:
        provenance_str = node.get("provenance", "[]")
        try:
            provenance = json.loads(provenance_str)
        except (json.JSONDecodeError, TypeError):
            provenance = []

        meta_str = node.get("metadata", "{}")
        try:
            metadata = json.loads(meta_str)
        except (json.JSONDecodeError, TypeError):
            metadata = {}

        program = self._graph_to_program(node.get("id", ""), node.get("version", 1))

        return DefenseProgramRecord(
            id=node.get("id", ""),
            name=node.get("name", ""),
            version=int(node.get("version", 1)),
            confidence=float(node.get("confidence", 0.0)),
            provenance=provenance,
            created_at=float(node.get("created_at", 0.0)),
            updated_at=float(node.get("updated_at", 0.0)),
            metadata=metadata,
            program=program,
        )

    def _graph_to_program(
        self, program_id: str, version: int
    ) -> Program:
        with self._session() as session:
            result = session.run(
                """MATCH (p:DefenseProgram {id: $pid, version: $ver})
                           -[:HAS_ROOT]->(root:ASTNode)
                   RETURN root""",
                pid=program_id,
                ver=version,
            )
            record = result.single()
            if record is None:
                raise ValueError(
                    f"No root AST node for program {program_id} v{version}"
                )

            root = self._graph_to_node(record["root"], session)
            if not isinstance(root, IfThenElseNode):
                raise ValueError(
                    f"Root AST node is {type(root).__name__}, expected IfThenElseNode"
                )

            return Program(
                root=root,
                id=program_id,
            )

    def _graph_to_node(self, node_data: Any, session: Session) -> Node:
        node_type = node_data.get("node_type", "")
        node_id = node_data.get("node_id", "")

        if node_type == "PredicateNode":
            return self._make_atomic_node(node_data, PredicateNode, session)
        elif node_type == "TransformNode":
            return self._make_atomic_node(node_data, TransformNode, session)
        elif node_type == "ClassifierNode":
            return self._make_atomic_node(node_data, ClassifierNode, session)
        elif node_type == "ThresholdNode":
            return self._make_threshold_node(node_data, session)
        elif node_type == "ApplyTransformNode":
            return self._make_apply_transform_node(node_data, session)
        elif node_type == "AndNode":
            return self._make_binary_node(node_data, AndNode, session)
        elif node_type == "OrNode":
            return self._make_binary_node(node_data, OrNode, session)
        elif node_type == "NotNode":
            return self._make_not_node(node_data, session)
        elif node_type == "IfThenElseNode":
            return self._make_ite_node(node_data, session)
        else:
            raise ValueError(f"Unknown AST node type: {node_type}")

    def _make_atomic_node(
        self, node_data: Any, node_cls: Type[AtomicNode], session: Session
    ) -> AtomicNode:
        node_id = node_data.get("node_id", "")
        params_str = node_data.get("parameters_json", "{}")
        try:
            parameters = json.loads(params_str)
        except (json.JSONDecodeError, TypeError):
            parameters = {}

        result = session.run(
            "MATCH (n:ASTNode {node_id: $nid})-[:PRIMITIVE]->(p:PrimitiveNode) "
            "RETURN p.name AS name, p.primitive_type AS ptype",
            nid=node_id,
        )
        record = result.single()
        if record is None:
            raise ValueError(f"No PrimitiveNode for ASTNode {node_id}")

        cls_map = {
            "PredicateNode": PredicateNode,
            "TransformNode": TransformNode,
            "ClassifierNode": ClassifierNode,
        }

        from core.primitive import PrimitiveRegistry

        registry = PrimitiveRegistry()
        try:
            primitive = registry.get(record["name"], parameters)
        except ValueError:
            from core.primitive import Predicate, Transform, Classifier

            primitive_cls: Any = {"Predicate": Predicate, "Transform": Transform, "Classifier": Classifier}.get(record["ptype"], Predicate)
            primitive = primitive_cls(name=record["name"], parameters=parameters)

        return node_cls(primitive=primitive)

    def _make_threshold_node(
        self, node_data: Any, session: Session
    ) -> ThresholdNode:
        node_id = node_data.get("node_id", "")
        threshold = float(node_data.get("threshold", 0.0))
        params_str = node_data.get("parameters_json", "{}")
        try:
            parameters = json.loads(params_str)
        except (json.JSONDecodeError, TypeError):
            parameters = {}

        result = session.run(
            "MATCH (n:ASTNode {node_id: $nid})-[:PRIMITIVE]->(p:PrimitiveNode) "
            "RETURN p.name AS name",
            nid=node_id,
        )
        record = result.single()
        if record is None:
            raise ValueError(f"No PrimitiveNode for ThresholdNode {node_id}")

        from core.primitive import PrimitiveRegistry, Classifier

        registry = PrimitiveRegistry()
        try:
            classifier = registry.get(record["name"], parameters)
        except ValueError:
            classifier = Classifier(name=record["name"], parameters=parameters)

        return ThresholdNode(classifier=classifier, threshold=threshold)

    def _make_apply_transform_node(
        self, node_data: Any, session: Session
    ) -> ApplyTransformNode:
        node_id = node_data.get("node_id", "")
        params_str = node_data.get("parameters_json", "{}")
        try:
            parameters = json.loads(params_str)
        except (json.JSONDecodeError, TypeError):
            parameters = {}

        result = session.run(
            "MATCH (n:ASTNode {node_id: $nid})-[:PRIMITIVE]->(p:PrimitiveNode) "
            "RETURN p.name AS name",
            nid=node_id,
        )
        record = result.single()
        if record is None:
            raise ValueError(f"No PrimitiveNode for ApplyTransformNode {node_id}")

        result2 = session.run(
            "MATCH (n:ASTNode {node_id: $nid})-[:INNER]->(inner:ASTNode) "
            "RETURN inner",
            nid=node_id,
        )
        inner_record = result2.single()
        if inner_record is None:
            raise ValueError(f"No INNER child for ApplyTransformNode {node_id}")

        inner = self._graph_to_node(inner_record["inner"], session)

        from core.primitive import PrimitiveRegistry, Transform

        registry = PrimitiveRegistry()
        try:
            transform = registry.get(record["name"], parameters)
        except ValueError:
            transform = Transform(name=record["name"], parameters=parameters)

        return ApplyTransformNode(transform=transform, inner=inner)

    def _make_binary_node(
        self, node_data: Any, node_cls: Type[BinaryNode], session: Session
    ) -> BinaryNode:
        node_id = node_data.get("node_id", "")

        left_result = session.run(
            "MATCH (n:ASTNode {node_id: $nid})-[:LEFT]->(left:ASTNode) RETURN left",
            nid=node_id,
        )
        left_record = left_result.single()
        if left_record is None:
            raise ValueError(f"No LEFT child for {node_data.get('node_type')} {node_id}")

        right_result = session.run(
            "MATCH (n:ASTNode {node_id: $nid})-[:RIGHT]->(right:ASTNode) RETURN right",
            nid=node_id,
        )
        right_record = right_result.single()
        if right_record is None:
            raise ValueError(f"No RIGHT child for {node_data.get('node_type')} {node_id}")

        left = self._graph_to_node(left_record["left"], session)
        right = self._graph_to_node(right_record["right"], session)

        if node_cls is AndNode:
            return AndNode(left=left, right=right)
        return OrNode(left=left, right=right)

    def _make_not_node(
        self, node_data: Any, session: Session
    ) -> NotNode:
        node_id = node_data.get("node_id", "")

        result = session.run(
            "MATCH (n:ASTNode {node_id: $nid})-[:CHILD]->(child:ASTNode) RETURN child",
            nid=node_id,
        )
        record = result.single()
        if record is None:
            raise ValueError(f"No CHILD for NotNode {node_id}")

        child = self._graph_to_node(record["child"], session)
        return NotNode(child=child)

    def _make_ite_node(
        self, node_data: Any, session: Session
    ) -> IfThenElseNode:
        node_id = node_data.get("node_id", "")
        then_outcome = int(node_data.get("then_outcome", 1))
        else_outcome = int(node_data.get("else_outcome", 0))

        result = session.run(
            "MATCH (n:ASTNode {node_id: $nid})-[:CONDITION]->(cond:ASTNode) RETURN cond",
            nid=node_id,
        )
        record = result.single()
        if record is None:
            raise ValueError(f"No CONDITION for IfThenElseNode {node_id}")

        condition = self._graph_to_node(record["cond"], session)
        return IfThenElseNode(
            condition=condition,
            then_outcome=then_outcome,
            else_outcome=else_outcome,
        )

    def _ast_to_graph(self, tx: Any, node: Node) -> str:
        prefix = f"ast_{uuid.uuid4().hex[:12]}"
        return self._create_ast_node(tx, node, prefix)

    def _create_ast_node(self, tx: Any, node: Node, prefix: str) -> str:
        node_id = f"{prefix}_{uuid.uuid4().hex[:8]}"

        base_props: Dict[str, Any] = {
            "node_id": node_id,
            "node_type": type(node).__name__,
        }

        if isinstance(node, ThresholdNode):
            base_props["threshold"] = node.threshold
            base_props["then_outcome"] = None
            base_props["else_outcome"] = None
            base_props["parameters_json"] = json.dumps(
                node.classifier.parameters, ensure_ascii=False
            )
        elif isinstance(node, IfThenElseNode):
            base_props["threshold"] = None
            base_props["then_outcome"] = node.then_outcome
            base_props["else_outcome"] = node.else_outcome
            base_props["parameters_json"] = "{}"
        elif isinstance(node, AtomicNode):
            base_props["threshold"] = None
            base_props["then_outcome"] = None
            base_props["else_outcome"] = None
            base_props["parameters_json"] = json.dumps(
                node.primitive.parameters, ensure_ascii=False
            )
        elif isinstance(node, ApplyTransformNode):
            base_props["threshold"] = None
            base_props["then_outcome"] = None
            base_props["else_outcome"] = None
            base_props["parameters_json"] = json.dumps(
                node.transform.parameters, ensure_ascii=False
            )
        else:
            base_props["threshold"] = None
            base_props["then_outcome"] = None
            base_props["else_outcome"] = None
            base_props["parameters_json"] = "{}"

        props_str = ", ".join(f"{k}: ${k}" for k in base_props)
        tx.run(f"CREATE (n:ASTNode {{{props_str}}})", **base_props)

        if isinstance(node, PredicateNode):
            self._ensure_and_link_primitive(
                tx, node_id, node.primitive, "Predicate"
            )
        elif isinstance(node, TransformNode):
            self._ensure_and_link_primitive(
                tx, node_id, node.primitive, "Transform"
            )
        elif isinstance(node, ClassifierNode):
            self._ensure_and_link_primitive(
                tx, node_id, node.primitive, "Classifier"
            )
        elif isinstance(node, ThresholdNode):
            self._ensure_and_link_primitive(
                tx, node_id, node.classifier, "Classifier"
            )
        elif isinstance(node, ApplyTransformNode):
            self._ensure_and_link_primitive(
                tx, node_id, node.transform, "Transform"
            )
            inner_id = self._create_ast_node(tx, node.inner, prefix)
            tx.run(
                "MATCH (n:ASTNode {node_id: $nid}), "
                "(c:ASTNode {node_id: $cid}) "
                "CREATE (n)-[:INNER]->(c)",
                nid=node_id, cid=inner_id,
            )
        elif isinstance(node, BinaryNode):
            left_id = self._create_ast_node(tx, node.left, prefix)
            right_id = self._create_ast_node(tx, node.right, prefix)
            tx.run(
                "MATCH (n:ASTNode {node_id: $nid}), "
                "(l:ASTNode {node_id: $lid}) "
                "CREATE (n)-[:LEFT]->(l)",
                nid=node_id, lid=left_id,
            )
            tx.run(
                "MATCH (n:ASTNode {node_id: $nid}), "
                "(r:ASTNode {node_id: $rid}) "
                "CREATE (n)-[:RIGHT]->(r)",
                nid=node_id, rid=right_id,
            )
        elif isinstance(node, NotNode):
            child_id = self._create_ast_node(tx, node.child, prefix)
            tx.run(
                "MATCH (n:ASTNode {node_id: $nid}), "
                "(c:ASTNode {node_id: $cid}) "
                "CREATE (n)-[:CHILD]->(c)",
                nid=node_id, cid=child_id,
            )
        elif isinstance(node, IfThenElseNode):
            cond_id = self._create_ast_node(tx, node.condition, prefix)
            tx.run(
                "MATCH (n:ASTNode {node_id: $nid}), "
                "(c:ASTNode {node_id: $cid}) "
                "CREATE (n)-[:CONDITION]->(c)",
                nid=node_id, cid=cond_id,
            )

        return node_id

    def _ensure_and_link_primitive(
        self,
        tx: Any,
        node_id: str,
        primitive: Any,
        primitive_type_label: str,
    ) -> None:
        pname = primitive.name
        tx.run(
            """MERGE (p:PrimitiveNode {name: $name})
               ON CREATE SET
                   p.primitive_type = $ptype,
                   p.parameters_schema = $pschema,
                   p.created_at = $created_at
               ON MATCH SET
                   p.primitive_type = $ptype
            """,
            name=pname,
            ptype=primitive_type_label,
            pschema=json.dumps(
                {k: _infer_type_of(v) for k, v in primitive.parameters.items()},
                ensure_ascii=False,
            ),
            created_at=__import__("time").time(),
        )
        tx.run(
            "MATCH (n:ASTNode {node_id: $nid}), "
            "(p:PrimitiveNode {name: $pname}) "
            "CREATE (n)-[:PRIMITIVE]->(p)",
            nid=node_id, pname=pname,
        )


def _infer_type_of(value: Any) -> str:
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "float"
    if isinstance(value, str):
        return "string"
    if isinstance(value, list):
        return "array"
    if isinstance(value, dict):
        return "object"
    return "string"
