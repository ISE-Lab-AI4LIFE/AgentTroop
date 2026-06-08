from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from neo4j import GraphDatabase, Session, basic_auth
from neo4j.exceptions import ServiceUnavailable

logger = logging.getLogger(__name__)


@dataclass
class OntologyPrimitive:
    name: str
    primitive_type: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    input_type: str = "String"
    output_type: str = "String"
    description: str = ""
    is_builtin: bool = False
    created_at: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        valid_types = {"predicate", "transform", "classifier", "policy"}
        if self.primitive_type not in valid_types:
            raise ValueError(
                f"Invalid primitive_type '{self.primitive_type}'. "
                f"Must be one of {valid_types}"
            )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "primitive_type": self.primitive_type,
            "parameters": self.parameters,
            "input_type": self.input_type,
            "output_type": self.output_type,
            "description": self.description,
            "is_builtin": self.is_builtin,
            "created_at": self.created_at,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "OntologyPrimitive":
        return cls(
            name=data["name"],
            primitive_type=data.get("primitive_type", "predicate"),
            parameters=dict(data.get("parameters", {})),
            input_type=data.get("input_type", "String"),
            output_type=data.get("output_type", "String"),
            description=data.get("description", ""),
            is_builtin=bool(data.get("is_builtin", False)),
            created_at=float(data.get("created_at", time.time())),
            metadata=dict(data.get("metadata", {})),
        )


class OntologyMemory:
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
                    self.uri, auth=basic_auth(self.user, self.password)
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

    def __enter__(self) -> "OntologyMemory":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    def _ensure_constraints(self) -> None:
        with self._session() as session:
            session.run(
                "CREATE CONSTRAINT IF NOT EXISTS "
                "FOR (p:OntologyPrimitive) REQUIRE p.name IS UNIQUE"
            )
            session.run(
                "CREATE INDEX IF NOT EXISTS FOR (p:OntologyPrimitive) "
                "ON (p.primitive_type)"
            )
            session.run(
                "CREATE INDEX IF NOT EXISTS FOR (p:OntologyPrimitive) "
                "ON (p.is_builtin)"
            )

    def save_primitive(self, primitive: OntologyPrimitive, overwrite: bool = False) -> str:
        if not overwrite:
            existing = self.get_primitive(primitive.name)
            if existing is not None:
                raise ValueError(
                    f"Primitive '{primitive.name}' already exists. "
                    "Use overwrite=True to replace."
                )

        with self._session() as session:
            session.run(
                """MERGE (p:OntologyPrimitive {name: $name})
                   SET p.primitive_type = $primitive_type,
                       p.parameters_json = $parameters_json,
                       p.input_type = $input_type,
                       p.output_type = $output_type,
                       p.description = $description,
                       p.is_builtin = $is_builtin,
                       p.created_at = $created_at,
                       p.metadata = $metadata
                """,
                name=primitive.name,
                primitive_type=primitive.primitive_type,
                parameters_json=json.dumps(primitive.parameters, ensure_ascii=False),
                input_type=primitive.input_type,
                output_type=primitive.output_type,
                description=primitive.description,
                is_builtin=primitive.is_builtin,
                created_at=primitive.created_at,
                metadata=json.dumps(primitive.metadata, ensure_ascii=False),
            )
        logger.debug("Saved ontology primitive '%s'", primitive.name)
        return primitive.name

    def get_primitive(self, name: str) -> Optional[OntologyPrimitive]:
        with self._session() as session:
            result = session.run(
                "MATCH (p:OntologyPrimitive {name: $name}) RETURN p",
                name=name,
            )
            record = result.single()
            if record is None:
                return None
            return self._node_to_primitive(record["p"])

    def delete_primitive(self, name: str) -> bool:
        with self._session() as session:
            result = session.run(
                "MATCH (p:OntologyPrimitive {name: $name}) "
                "DETACH DELETE p "
                "RETURN count(p) AS deleted",
                name=name,
            )
            record = result.single()
            return record is not None and record["deleted"] > 0

    def delete_all(self) -> int:
        with self._session() as session:
            result = session.run(
                "MATCH (p:OntologyPrimitive) DETACH DELETE p "
                "RETURN count(p) AS deleted"
            )
            record = result.single()
            return record["deleted"] if record else 0

    def list_primitives(
        self, primitive_type: Optional[str] = None
    ) -> List[OntologyPrimitive]:
        if primitive_type:
            with self._session() as session:
                result = session.run(
                    "MATCH (p:OntologyPrimitive {primitive_type: $ptype}) "
                    "RETURN p ORDER BY p.name",
                    ptype=primitive_type,
                )
                return [self._node_to_primitive(rec["p"]) for rec in result]
        else:
            with self._session() as session:
                result = session.run(
                    "MATCH (p:OntologyPrimitive) RETURN p ORDER BY p.name"
                )
                return [self._node_to_primitive(rec["p"]) for rec in result]

    def find_primitives(
        self,
        name_contains: Optional[str] = None,
        primitive_type: Optional[str] = None,
    ) -> List[OntologyPrimitive]:
        where_clauses: List[str] = []
        params: Dict[str, Any] = {}

        if name_contains:
            where_clauses.append("toLower(p.name) CONTAINS toLower($name_substr)")
            params["name_substr"] = name_contains
        if primitive_type:
            where_clauses.append("p.primitive_type = $ptype")
            params["ptype"] = primitive_type

        where_str = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""

        with self._session() as session:
            result = session.run(
                f"MATCH (p:OntologyPrimitive) {where_str} RETURN p ORDER BY p.name",
                **params,
            )
            return [self._node_to_primitive(rec["p"]) for rec in result]

    def sync_to_registry(self, overwrite: bool = False) -> int:
        from core.primitive import default_registry

        type_map: Dict[str, str] = {
            "Predicate": "predicate",
            "Transform": "transform",
            "Classifier": "classifier",
        }

        name_list = default_registry.list_primitives()
        count = 0
        for name in name_list:
            try:
                instance = default_registry.get(name)
            except ValueError:
                continue

            cls_name = type(instance).__name__
            ptype = type_map.get(cls_name, "predicate")

            op = OntologyPrimitive(
                name=instance.name,
                primitive_type=ptype,
                parameters={
                    k: _infer_type(v)
                    for k, v in instance.parameters.items()
                },
                input_type=instance.input_type,
                output_type=instance.output_type,
                description=instance.metadata.get("description", ""),
                is_builtin=True,
            )
            try:
                self.save_primitive(op, overwrite=overwrite)
                count += 1
            except ValueError:
                continue

        logger.info("Synced %d primitives from registry", count)
        return count

    def export_primitives(self, file_path: str) -> None:
        primitives = self.list_primitives()
        data = [p.to_dict() for p in primitives]
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        logger.info("Exported %d primitives to %s", len(primitives), file_path)

    def import_primitives(
        self, file_path: str, overwrite: bool = False
    ) -> int:
        with open(file_path, "r", encoding="utf-8") as f:
            raw_list = json.load(f)

        count = 0
        for raw in raw_list:
            primitive = OntologyPrimitive.from_dict(raw)
            try:
                self.save_primitive(primitive, overwrite=overwrite)
                count += 1
            except ValueError:
                continue
        logger.info("Imported %d primitives from %s", count, file_path)
        return count

    @staticmethod
    def _node_to_primitive(node: Any) -> OntologyPrimitive:
        params_str = node.get("parameters_json", "{}")
        try:
            parameters = json.loads(params_str)
        except (json.JSONDecodeError, TypeError):
            parameters = {}

        meta_str = node.get("metadata", "{}")
        try:
            metadata = json.loads(meta_str)
        except (json.JSONDecodeError, TypeError):
            metadata = {}

        return OntologyPrimitive(
            name=node.get("name", ""),
            primitive_type=node.get("primitive_type", "predicate"),
            parameters=parameters,
            input_type=node.get("input_type", "String"),
            output_type=node.get("output_type", "String"),
            description=node.get("description", ""),
            is_builtin=bool(node.get("is_builtin", False)),
            created_at=float(node.get("created_at", time.time())),
            metadata=metadata,
        )


def _infer_type(value: Any) -> str:
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
