"""Tests for Ontology Memory (L5 — Neo4j primitive catalog)."""

import json
import os
import tempfile
import time
from typing import Any, Dict

import pytest

from knowledge.ontology_memory import OntologyMemory, OntologyPrimitive

NEO4J_URI = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.environ.get("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD", "password")


def _neo4j_reachable() -> bool:
    try:
        m = OntologyMemory(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
        m._ensure_constraints()
        m.close()
        return True
    except Exception:
        return False


neo4j_required = pytest.mark.skipif(
    not _neo4j_reachable(),
    reason="Neo4j is not reachable — start it with: "
    "docker run --rm -p 7687:7687 -e NEO4J_AUTH=neo4j/password neo4j:5",
)


@pytest.fixture(scope="module")
def memory() -> OntologyMemory:
    m = OntologyMemory(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
    m._ensure_constraints()
    m.delete_all()
    yield m
    m.delete_all()
    m.close()


def make_prim(**overrides: Any) -> OntologyPrimitive:
    base: Dict[str, Any] = dict(
        name="test_predicate",
        primitive_type="predicate",
        parameters={"threshold": "integer"},
        input_type="String",
        output_type="Boolean",
        description="A test predicate",
        is_builtin=False,
    )
    base.update(overrides)
    return OntologyPrimitive(**base)


class TestOntologyPrimitiveDataclass:
    def test_default_values(self) -> None:
        p = OntologyPrimitive(name="p1", primitive_type="predicate")
        assert p.name == "p1"
        assert p.primitive_type == "predicate"
        assert p.parameters == {}
        assert p.input_type == "String"
        assert p.output_type == "String"
        assert p.description == ""
        assert p.is_builtin is False

    def test_validates_primitive_type(self) -> None:
        with pytest.raises(ValueError, match="Invalid primitive_type"):
            OntologyPrimitive(name="bad", primitive_type="unknown")

    def test_accepts_all_valid_types(self) -> None:
        for t in ("predicate", "transform", "classifier", "policy"):
            p = OntologyPrimitive(name=f"p_{t}", primitive_type=t)
            assert p.primitive_type == t

    def test_to_dict_roundtrip(self) -> None:
        p1 = make_prim(
            name="contains_word",
            primitive_type="predicate",
            parameters={"word": "string"},
            description="Check word containment",
            metadata={"category": "text"},
        )
        d = p1.to_dict()
        p2 = OntologyPrimitive.from_dict(d)
        assert p2.name == p1.name
        assert p2.primitive_type == p1.primitive_type
        assert p2.parameters == p1.parameters
        assert p2.description == p1.description
        assert p2.metadata == p1.metadata

    def test_from_dict_minimal(self) -> None:
        raw = {"name": "minimal", "primitive_type": "classifier"}
        p = OntologyPrimitive.from_dict(raw)
        assert p.name == "minimal"
        assert p.primitive_type == "classifier"
        assert p.parameters == {}
        assert p.input_type == "String"


class TestOntologyMemoryNeo4j:
    @neo4j_required
    def test_save_and_get_primitive(self, memory: OntologyMemory) -> None:
        p = make_prim(name="save_get_test")
        saved = memory.save_primitive(p)
        assert saved == "save_get_test"

        retrieved = memory.get_primitive("save_get_test")
        assert retrieved is not None
        assert retrieved.name == "save_get_test"
        assert retrieved.primitive_type == "predicate"

    @neo4j_required
    def test_save_duplicate_raises(self, memory: OntologyMemory) -> None:
        p = make_prim(name="dup_test")
        memory.save_primitive(p)
        with pytest.raises(ValueError, match="already exists"):
            memory.save_primitive(p)

    @neo4j_required
    def test_save_duplicate_with_overwrite(self, memory: OntologyMemory) -> None:
        p1 = make_prim(name="overwrite_test", description="original")
        memory.save_primitive(p1)
        p2 = make_prim(name="overwrite_test", description="overwritten")
        saved = memory.save_primitive(p2, overwrite=True)
        assert saved == "overwrite_test"

        retrieved = memory.get_primitive("overwrite_test")
        assert retrieved is not None
        assert retrieved.description == "overwritten"

    @neo4j_required
    def test_get_nonexistent(self, memory: OntologyMemory) -> None:
        assert memory.get_primitive("nonexistent_12345") is None

    @neo4j_required
    def test_delete_primitive(self, memory: OntologyMemory) -> None:
        p = make_prim(name="delete_me")
        memory.save_primitive(p)
        assert memory.delete_primitive("delete_me") is True
        assert memory.get_primitive("delete_me") is None

    @neo4j_required
    def test_delete_nonexistent(self, memory: OntologyMemory) -> None:
        assert memory.delete_primitive("no_such_primitive") is False

    @neo4j_required
    def test_list_all_primitives(self, memory: OntologyMemory) -> None:
        memory.save_primitive(make_prim(name="list_a", primitive_type="predicate"))
        memory.save_primitive(make_prim(name="list_b", primitive_type="transform"))
        all_ps = memory.list_primitives()
        names = [p.name for p in all_ps]
        assert "list_a" in names
        assert "list_b" in names

    @neo4j_required
    def test_list_by_type(self, memory: OntologyMemory) -> None:
        memory.save_primitive(make_prim(name="type_pred_a", primitive_type="predicate"))
        memory.save_primitive(make_prim(name="type_trans_b", primitive_type="transform"))
        preds = memory.list_primitives(primitive_type="predicate")
        assert all(p.primitive_type == "predicate" for p in preds)
        assert any(p.name == "type_pred_a" for p in preds)
        assert not any(p.name == "type_trans_b" for p in preds)

    @neo4j_required
    def test_find_by_name_contains(self, memory: OntologyMemory) -> None:
        memory.save_primitive(make_prim(name="find_hello_world"))
        memory.save_primitive(make_prim(name="find_hello_there"))
        memory.save_primitive(make_prim(name="other_thing"))
        results = memory.find_primitives(name_contains="hello")
        assert len(results) == 2
        assert all("hello" in p.name for p in results)

    @neo4j_required
    def test_find_by_name_and_type(self, memory: OntologyMemory) -> None:
        memory.save_primitive(make_prim(name="combo_a", primitive_type="predicate"))
        memory.save_primitive(make_prim(name="combo_b", primitive_type="classifier"))
        results = memory.find_primitives(name_contains="combo", primitive_type="predicate")
        assert len(results) == 1
        assert results[0].name == "combo_a"

    @neo4j_required
    def test_sync_to_registry(self, memory: OntologyMemory) -> None:
        count = memory.sync_to_registry(overwrite=True)
        assert count > 0

        all_ps = memory.list_primitives()
        names = [p.name for p in all_ps]
        assert "contains_word" in names
        assert "rot13" in names
        assert "toxicity_score" in names

        for p in all_ps:
            if p.is_builtin:
                assert p.primitive_type in ("predicate", "transform", "classifier")

    @neo4j_required
    def test_export_import_primitives(self, memory: OntologyMemory) -> None:
        memory.save_primitive(make_prim(name="export_me"))
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            exp_path = f.name

        try:
            memory.export_primitives(exp_path)
            with open(exp_path, "r") as f:
                data = json.load(f)
            exported_names = [d["name"] for d in data]
            assert "export_me" in exported_names

            memory.delete_primitive("export_me")
            count = memory.import_primitives(exp_path)
            assert count >= 1
            assert memory.get_primitive("export_me") is not None
        finally:
            if os.path.exists(exp_path):
                os.unlink(exp_path)

    @neo4j_required
    def test_delete_all(self, memory: OntologyMemory) -> None:
        memory.save_primitive(make_prim(name="clean_a"))
        memory.save_primitive(make_prim(name="clean_b"))
        deleted = memory.delete_all()
        assert deleted >= 2
        assert memory.list_primitives() == []

    @neo4j_required
    def test_context_manager(self) -> None:
        with OntologyMemory(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD) as m:
            m._ensure_constraints()
            assert m._driver is not None
        assert m._driver is None
