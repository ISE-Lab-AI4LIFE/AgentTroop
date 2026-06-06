"""Tests for Defense Program Store (L4 — Neo4j program AST store)."""

import json
import os
import tempfile
import time
from typing import Any, Dict

import pytest

from core.program import (
    AndNode,
    ApplyTransformNode,
    ClassifierNode,
    IfThenElseNode,
    NotNode,
    OrNode,
    PredicateNode,
    Program,
    ThresholdNode,
    TransformNode,
)
from core.primitive import (
    Base64DecodeTransform,
    Classifier,
    ContainsWordPredicate,
    LengthGtPredicate,
    Predicate,
    Rot13Transform,
    ToxicityScoreClassifier,
    Transform,
)
from knowledge.defense_store import DefenseProgramRecord, DefenseProgramStore

NEO4J_URI = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.environ.get("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD", "password")


def _neo4j_reachable() -> bool:
    try:
        m = DefenseProgramStore(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
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
def store() -> DefenseProgramStore:
    s = DefenseProgramStore(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
    s._ensure_constraints()
    s.delete_all()
    yield s
    s.delete_all()
    s.close()


def simple_program() -> Program:
    return Program(
        root=IfThenElseNode(
            condition=PredicateNode(
                primitive=ContainsWordPredicate(word="bomb")
            ),
            then_outcome=1,
            else_outcome=0,
        ),
    )


def complex_program() -> Program:
    return Program(
        root=IfThenElseNode(
            condition=AndNode(
                left=PredicateNode(
                    primitive=ContainsWordPredicate(word="bomb")
                ),
                right=NotNode(
                    child=ApplyTransformNode(
                        transform=Rot13Transform(),
                        inner=ThresholdNode(
                            classifier=ToxicityScoreClassifier(),
                            threshold=0.5,
                        ),
                    )
                ),
            ),
            then_outcome=1,
            else_outcome=0,
        ),
    )


def make_record(**overrides: Any) -> DefenseProgramRecord:
    base: Dict[str, Any] = dict(
        name="test_program",
        program=simple_program(),
        confidence=0.85,
        provenance=["ep_001", "ep_002"],
    )
    base.update(overrides)
    return DefenseProgramRecord(**base)


class TestDefenseProgramRecordDataclass:
    def test_auto_generates_id(self) -> None:
        r = DefenseProgramRecord(name="test", program=simple_program())
        assert r.id.startswith("dfp_")
        assert len(r.id) > 4

    def test_clamps_confidence(self) -> None:
        r = DefenseProgramRecord(name="x", program=simple_program(), confidence=1.5)
        assert r.confidence == 1.0
        r2 = DefenseProgramRecord(name="x", program=simple_program(), confidence=-0.5)
        assert r2.confidence == 0.0

    def test_default_version(self) -> None:
        r = DefenseProgramRecord(name="x", program=simple_program())
        assert r.version == 1

    def test_clamps_version(self) -> None:
        r = DefenseProgramRecord(
            name="x", program=simple_program(), version=0
        )
        assert r.version == 1

    def test_to_dict_roundtrip(self) -> None:
        r1 = DefenseProgramRecord(
            name="roundtrip",
            program=complex_program(),
            confidence=0.9,
            provenance=["ep_001"],
        )
        d = r1.to_dict()
        r2 = DefenseProgramRecord.from_dict(d)
        assert r2.name == "roundtrip"
        assert r2.confidence == 0.9
        assert r2.program.root.condition == r1.program.root.condition

    def test_from_dict_minimal(self) -> None:
        raw = {
            "name": "minimal",
            "program": simple_program().to_dict(),
        }
        r = DefenseProgramRecord.from_dict(raw)
        assert r.name == "minimal"
        assert r.version == 1
        assert r.program is not None


class TestDefenseProgramStoreNeo4j:
    @neo4j_required
    def test_save_and_get(self, store: DefenseProgramStore) -> None:
        r = make_record(name="save_get_test")
        pid = store.save(r)
        assert pid == r.id

        retrieved = store.get(pid)
        assert retrieved is not None
        assert retrieved.name == "save_get_test"
        assert retrieved.confidence == 0.85

    @neo4j_required
    def test_get_nonexistent(self, store: DefenseProgramStore) -> None:
        assert store.get("nonexistent_12345") is None

    @neo4j_required
    def test_versioning(self, store: DefenseProgramStore) -> None:
        r1 = make_record(name="ver_test", confidence=0.7)
        pid = store.save(r1)

        r2 = make_record(name="ver_test", confidence=0.9, id=pid)
        store.save(r2)

        v1 = store.get(pid, version=1)
        assert v1 is not None
        assert v1.confidence == 0.7

        v2 = store.get(pid, version=2)
        assert v2 is not None
        assert v2.confidence == 0.9

        latest = store.get(pid)
        assert latest is not None
        assert latest.version == 2

    @neo4j_required
    def test_get_specific_version_nonexistent(
        self, store: DefenseProgramStore
    ) -> None:
        r = make_record(name="ver_miss")
        pid = store.save(r)
        assert store.get(pid, version=999) is None

    @neo4j_required
    def test_delete_program(self, store: DefenseProgramStore) -> None:
        r = make_record(name="delete_test")
        pid = store.save(r)
        assert store.delete(pid) is True
        assert store.get(pid) is None

    @neo4j_required
    def test_delete_nonexistent(self, store: DefenseProgramStore) -> None:
        assert store.delete("no_such_program") is False

    @neo4j_required
    def test_find_by_confidence(self, store: DefenseProgramStore) -> None:
        store.save(make_record(name="high_conf", confidence=0.95))
        store.save(make_record(name="low_conf", confidence=0.3))

        high = store.find_by_confidence(min_confidence=0.8)
        assert any(r.name == "high_conf" for r in high)
        assert not any(r.name == "low_conf" for r in high)

    @neo4j_required
    def test_find_by_primitive(self, store: DefenseProgramStore) -> None:
        store.save(make_record(name="find_prim_test"))

        results = store.find_by_primitive("contains_word")
        assert any(r.name == "find_prim_test" for r in results)

        results_none = store.find_by_primitive("nonexistent_primitive")
        assert len(results_none) == 0

    @neo4j_required
    def test_find_by_primitive_latest_only(self, store: DefenseProgramStore) -> None:
        r1 = make_record(name="prim_ver", confidence=0.5)
        pid = store.save(r1)
        r2 = make_record(name="prim_ver", confidence=0.9, id=pid)
        store.save(r2)

        results = store.find_by_primitive("contains_word")
        matches = [r for r in results if r.name == "prim_ver"]
        assert len(matches) == 1
        assert matches[0].version == 2

    @neo4j_required
    def test_list_program_ids(self, store: DefenseProgramStore) -> None:
        r1 = make_record(name="list_me_a")
        r2 = make_record(name="list_me_b")
        pid1 = store.save(r1)
        pid2 = store.save(r2)

        ids = store.list_program_ids()
        assert pid1 in ids
        assert pid2 in ids

    @neo4j_required
    def test_update_confidence(self, store: DefenseProgramStore) -> None:
        r = make_record(name="update_conf", confidence=0.5)
        pid = store.save(r)

        assert store.update_confidence(pid, 0.95) is True
        updated = store.get(pid)
        assert updated is not None
        assert updated.confidence == 0.95
        assert updated.version == 2

    @neo4j_required
    def test_update_confidence_nonexistent(
        self, store: DefenseProgramStore
    ) -> None:
        assert store.update_confidence("no_such_id", 0.9) is False

    @neo4j_required
    def test_complex_program_roundtrip(
        self, store: DefenseProgramStore
    ) -> None:
        prog = complex_program()
        r = make_record(name="complex_test", program=prog)
        pid = store.save(r)

        retrieved = store.get(pid)
        assert retrieved is not None
        assert retrieved.program.root == prog.root

    @neo4j_required
    def test_not_node_roundtrip(self, store: DefenseProgramStore) -> None:
        prog = Program(
            root=IfThenElseNode(
                condition=NotNode(
                    child=PredicateNode(
                        primitive=LengthGtPredicate(threshold=100)
                    )
                ),
                then_outcome=1,
                else_outcome=0,
            ),
        )
        r = make_record(name="not_test", program=prog)
        pid = store.save(r)

        retrieved = store.get(pid)
        assert retrieved is not None
        assert retrieved.program.root == prog.root

    @neo4j_required
    def test_apply_transform_threshold_roundtrip(
        self, store: DefenseProgramStore
    ) -> None:
        prog = Program(
            root=IfThenElseNode(
                condition=ApplyTransformNode(
                    transform=Base64DecodeTransform(),
                    inner=ThresholdNode(
                        classifier=ToxicityScoreClassifier(),
                        threshold=0.8,
                    ),
                ),
                then_outcome=1,
                else_outcome=0,
            ),
        )
        r = make_record(name="apply_thresh_test", program=prog)
        pid = store.save(r)

        retrieved = store.get(pid)
        assert retrieved is not None
        assert retrieved.program.root == prog.root

    @neo4j_required
    def test_export_import_programs(self, store: DefenseProgramStore) -> None:
        r = make_record(name="export_import_test")
        pid = store.save(r)

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            exp_path = f.name

        try:
            store.export_programs(exp_path)
            with open(exp_path, "r") as f:
                data = json.load(f)
            exported_ids = [d["id"] for d in data]
            assert pid in exported_ids

            store.delete(pid)
            count = store.import_programs(exp_path)
            assert count >= 1
            assert store.get(pid) is not None
        finally:
            if os.path.exists(exp_path):
                os.unlink(exp_path)

    @neo4j_required
    def test_export_latest_only(self, store: DefenseProgramStore) -> None:
        r1 = make_record(name="export_latest", confidence=0.5)
        pid = store.save(r1)
        r2 = make_record(name="export_latest", confidence=0.99, id=pid)
        store.save(r2)

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            exp_path = f.name

        try:
            store.export_programs(exp_path, include_history=False)
            with open(exp_path, "r") as f:
                data = json.load(f)
            my_entries = [d for d in data if d["id"] == pid]
            assert len(my_entries) == 1
            assert my_entries[0]["version"] == 2
        finally:
            if os.path.exists(exp_path):
                os.unlink(exp_path)

    @neo4j_required
    def test_delete_all(self, store: DefenseProgramStore) -> None:
        store.save(make_record(name="cleanup_a"))
        store.save(make_record(name="cleanup_b"))
        deleted = store.delete_all()
        assert deleted >= 2
        assert store.list_program_ids() == []

    @neo4j_required
    def test_context_manager(self) -> None:
        with DefenseProgramStore(
            NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD
        ) as s:
            s._ensure_constraints()
            assert s._driver is not None
        assert s._driver is None
