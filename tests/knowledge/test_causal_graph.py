"""Tests for CausalGraph (Neo4j-backed causal graph store)."""

import json
import os
import tempfile
import time
from typing import Any, Dict, Generator, List

import pytest

from knowledge.causal_graph import CausalEdge, CausalGraph, CausalNode

pytest.importorskip("testcontainers.neo4j")
from testcontainers.neo4j import Neo4jContainer  # type: ignore[import-untyped]

NEO4J_IMAGE = os.environ.get("NEO4J_IMAGE", "neo4j:5")


@pytest.fixture(scope="module")
def neo4j_params() -> Generator[Dict[str, Any], None, None]:
    container = Neo4jContainer(image=NEO4J_IMAGE)
    container.start()
    time.sleep(2)
    try:
        yield {
            "uri": container.get_connection_url(),
            "user": "neo4j",
            "password": container.password,
        }
    finally:
        container.stop()


@pytest.fixture
def graph(neo4j_params: Dict[str, Any]) -> Generator[CausalGraph, None, None]:
    g = CausalGraph(
        uri=neo4j_params["uri"],
        user=neo4j_params["user"],
        password=neo4j_params["password"],
    )
    g.clear()
    yield g
    g.clear()
    g.close()


# ---------------------------------------------------------------------------
# Dataclass unit tests
# ---------------------------------------------------------------------------


class TestCausalNode:
    def test_auto_generates_id(self) -> None:
        n = CausalNode(name="test")
        assert n.id.startswith("cnd_")
        assert len(n.id) > 4

    def test_keeps_provided_id(self) -> None:
        n = CausalNode(id="custom_id", name="x")
        assert n.id == "custom_id"

    def test_to_dict_roundtrip(self) -> None:
        n1 = CausalNode(
            name="ROT13",
            type="transform",
            metadata={"category": "cipher"},
        )
        data = n1.to_dict()
        n2 = CausalNode.from_dict(data)
        assert n1.id == n2.id
        assert n1.name == n2.name
        assert n1.type == n2.type
        assert n1.metadata == n2.metadata

    def test_default_type(self) -> None:
        n = CausalNode(name="x")
        assert n.type == "primitive"


class TestCausalEdge:
    def test_clamps_strength(self) -> None:
        e = CausalEdge(source_id="a", target_id="b", strength=1.5)
        assert e.strength == 1.0
        e2 = CausalEdge(source_id="a", target_id="b", strength=-0.5)
        assert e2.strength == 0.0

    def test_clamps_p_value(self) -> None:
        e = CausalEdge(source_id="a", target_id="b", p_value=-0.1)
        assert e.p_value == 0.0

    def test_to_dict_roundtrip(self) -> None:
        e1 = CausalEdge(
            source_id="a",
            target_id="b",
            strength=0.85,
            p_value=0.01,
            intervention_ids=["int_1", "int_2"],
        )
        data = e1.to_dict()
        e2 = CausalEdge.from_dict(data)
        for attr in ("source_id", "target_id", "strength", "p_value"):
            assert getattr(e1, attr) == getattr(e2, attr)
        assert e2.intervention_ids == ["int_1", "int_2"]


# ---------------------------------------------------------------------------
# Integration tests (Neo4j via testcontainers)
# ---------------------------------------------------------------------------


class TestCausalGraphNodeCRUD:
    def test_add_node(self, graph: CausalGraph) -> None:
        nid = graph.add_node("ROT13", "transform", {"desc": "ROT13 cipher"})
        assert nid.startswith("cnd_")
        node = graph.get_node(nid)
        assert node is not None
        assert node.name == "ROT13"
        assert node.type == "transform"
        assert node.metadata == {"desc": "ROT13 cipher"}

    def test_add_node_default_metadata(self, graph: CausalGraph) -> None:
        nid = graph.add_node("KeywordFilter", "defense_component")
        node = graph.get_node(nid)
        assert node is not None
        assert node.metadata == {}

    def test_get_node_nonexistent(self, graph: CausalGraph) -> None:
        assert graph.get_node("nonexistent") is None

    def test_get_or_create_node_new(self, graph: CausalGraph) -> None:
        nid = graph.get_or_create_node("ROT13", "transform")
        assert nid.startswith("cnd_")

    def test_get_or_create_node_existing(self, graph: CausalGraph) -> None:
        nid1 = graph.add_node("ROT13", "transform")
        nid2 = graph.get_or_create_node("ROT13", "transform")
        assert nid1 == nid2

    def test_update_node_name(self, graph: CausalGraph) -> None:
        nid = graph.add_node("OldName", "primitive")
        assert graph.update_node(nid, name="NewName")
        node = graph.get_node(nid)
        assert node is not None
        assert node.name == "NewName"

    def test_update_node_metadata(self, graph: CausalGraph) -> None:
        nid = graph.add_node("Test", "primitive")
        assert graph.update_node(nid, metadata={"key": "value"})
        node = graph.get_node(nid)
        assert node is not None
        assert node.metadata == {"key": "value"}

    def test_update_node_noop(self, graph: CausalGraph) -> None:
        nid = graph.add_node("Test", "primitive")
        assert not graph.update_node(nid)

    def test_delete_node(self, graph: CausalGraph) -> None:
        nid = graph.add_node("ToDelete", "primitive")
        assert graph.delete_node(nid)
        assert graph.get_node(nid) is None

    def test_delete_node_nonexistent(self, graph: CausalGraph) -> None:
        assert not graph.delete_node("nonexistent")

    def test_find_nodes_by_name(self, graph: CausalGraph) -> None:
        graph.add_node("ROT13", "transform")
        results = graph.find_nodes_by_name("ROT13")
        assert len(results) == 1
        assert results[0].name == "ROT13"

    def test_find_nodes_by_name_empty(self, graph: CausalGraph) -> None:
        assert graph.find_nodes_by_name("Nope") == []

    def test_find_nodes_by_type(self, graph: CausalGraph) -> None:
        graph.add_node("A", "transform")
        graph.add_node("B", "transform")
        graph.add_node("C", "outcome")
        results = graph.find_nodes_by_type("transform")
        assert len(results) == 2

    def test_get_all_nodes(self, graph: CausalGraph) -> None:
        graph.add_node("A", "primitive")
        graph.add_node("B", "outcome")
        nodes = graph.get_all_nodes()
        assert len(nodes) >= 2


class TestCausalGraphEdgeCRUD:
    def test_add_edge(self, graph: CausalGraph) -> None:
        sid = graph.add_node("Source", "primitive")
        tid = graph.add_node("Target", "outcome")
        assert graph.add_edge(sid, tid, 0.8, 0.01, ["int_1"])

    def test_get_edge(self, graph: CausalGraph) -> None:
        sid = graph.add_node("Source", "primitive")
        tid = graph.add_node("Target", "outcome")
        graph.add_edge(sid, tid, 0.75, 0.02, ["int_1"])
        edge = graph.get_edge(sid, tid)
        assert edge is not None
        assert edge.source_id == sid
        assert edge.target_id == tid
        assert edge.strength == 0.75
        assert edge.p_value == 0.02
        assert edge.intervention_ids == ["int_1"]

    def test_get_edge_nonexistent(self, graph: CausalGraph) -> None:
        assert graph.get_edge("a", "b") is None

    def test_delete_edge(self, graph: CausalGraph) -> None:
        sid = graph.add_node("S", "primitive")
        tid = graph.add_node("T", "outcome")
        graph.add_edge(sid, tid, 0.5, 0.05, [])
        assert graph.delete_edge(sid, tid)
        assert graph.get_edge(sid, tid) is None

    def test_delete_edge_nonexistent(self, graph: CausalGraph) -> None:
        assert not graph.delete_edge("a", "b")

    def test_add_edge_clamps_values(self, graph: CausalGraph) -> None:
        sid = graph.add_node("S", "primitive")
        tid = graph.add_node("T", "outcome")
        graph.add_edge(sid, tid, 1.5, -0.1, [])
        edge = graph.get_edge(sid, tid)
        assert edge is not None
        assert edge.strength == 1.0
        assert edge.p_value == 0.0

    def test_get_all_edges(self, graph: CausalGraph) -> None:
        s1 = graph.add_node("S1", "primitive")
        t1 = graph.add_node("T1", "outcome")
        s2 = graph.add_node("S2", "primitive")
        t2 = graph.add_node("T2", "outcome")
        graph.add_edge(s1, t1, 0.5, 0.05, [])
        graph.add_edge(s2, t2, 0.9, 0.01, [])
        edges = graph.get_all_edges()
        assert len(edges) == 2


class TestCausalGraphQuery:
    def test_find_causes(self, graph: CausalGraph) -> None:
        s1 = graph.add_node("Cause1", "primitive")
        s2 = graph.add_node("Cause2", "primitive")
        t = graph.add_node("Effect", "outcome")
        graph.add_edge(s1, t, 0.9, 0.01, [])
        graph.add_edge(s2, t, 0.3, 0.05, [])
        causes = graph.find_causes(t)
        assert len(causes) == 2
        assert causes[0][0].strength >= causes[1][0].strength

    def test_find_causes_with_min_strength(self, graph: CausalGraph) -> None:
        s1 = graph.add_node("Strong", "primitive")
        s2 = graph.add_node("Weak", "primitive")
        t = graph.add_node("Effect", "outcome")
        graph.add_edge(s1, t, 0.9, 0.01, [])
        graph.add_edge(s2, t, 0.2, 0.05, [])
        causes = graph.find_causes(t, min_strength=0.5)
        assert len(causes) == 1
        assert causes[0][0].strength == 0.9

    def test_find_effects(self, graph: CausalGraph) -> None:
        s = graph.add_node("Source", "primitive")
        t1 = graph.add_node("Effect1", "outcome")
        t2 = graph.add_node("Effect2", "outcome")
        graph.add_edge(s, t1, 0.8, 0.01, [])
        graph.add_edge(s, t2, 0.6, 0.03, [])
        effects = graph.find_effects(s)
        assert len(effects) == 2

    def test_find_effects_with_min_strength(self, graph: CausalGraph) -> None:
        s = graph.add_node("Source", "primitive")
        t1 = graph.add_node("Effect1", "outcome")
        t2 = graph.add_node("Effect2", "outcome")
        graph.add_edge(s, t1, 0.8, 0.01, [])
        graph.add_edge(s, t2, 0.3, 0.05, [])
        effects = graph.find_effects(s, min_strength=0.5)
        assert len(effects) == 1

    def test_find_causes_empty(self, graph: CausalGraph) -> None:
        nid = graph.add_node("Orphan", "primitive")
        assert graph.find_causes(nid) == []

    def test_find_effects_empty(self, graph: CausalGraph) -> None:
        nid = graph.add_node("Orphan", "primitive")
        assert graph.find_effects(nid) == []


class TestCausalGraphClearExportImport:
    def test_clear(self, graph: CausalGraph) -> None:
        graph.add_node("A", "primitive")
        graph.add_node("B", "primitive")
        assert graph.clear() >= 2
        assert graph.get_all_nodes() == []

    def test_export_import(self, graph: CausalGraph) -> None:
        s = graph.add_node("Cause", "primitive")
        t = graph.add_node("Effect", "outcome")
        graph.add_edge(s, t, 0.85, 0.01, ["int_1"])

        with tempfile.NamedTemporaryFile(
            suffix=".json", mode="w", delete=False
        ) as f:
            path = f.name
            graph.export(path)

        g2 = CausalGraph(
            uri=graph.uri,
            user=graph.user,
            password=graph.password,
        )
        g2.clear()
        try:
            count = g2.import_(path)
            assert count >= 2
            assert len(g2.get_all_nodes()) >= 2
            assert len(g2.get_all_edges()) >= 1
        finally:
            g2.clear()
            g2.close()
            os.unlink(path)

    def test_export_empty(self, graph: CausalGraph) -> None:
        with tempfile.NamedTemporaryFile(
            suffix=".json", mode="w", delete=False
        ) as f:
            path = f.name
            graph.export(path)

        with open(path, "r") as f:
            data = json.load(f)
        assert data["nodes"] == []
        assert data["edges"] == []
        os.unlink(path)

    def test_import_empty(self, graph: CausalGraph) -> None:
        with tempfile.NamedTemporaryFile(
            suffix=".json", mode="w", delete=False
        ) as f:
            json.dump({"nodes": [], "edges": []}, f)
            path = f.name

        count = graph.import_(path)
        assert count == 0
        os.unlink(path)

    def test_context_manager(self, neo4j_params: Dict[str, Any]) -> None:
        with CausalGraph(
            uri=neo4j_params["uri"],
            user=neo4j_params["user"],
            password=neo4j_params["password"],
        ) as g:
            nid = g.add_node("Test", "primitive")
            assert g.get_node(nid) is not None
        assert g._driver is None


class TestCausalGraphEdgeCases:
    def test_delete_node_cascades_edges(self, graph: CausalGraph) -> None:
        s = graph.add_node("Source", "primitive")
        t = graph.add_node("Target", "outcome")
        graph.add_edge(s, t, 0.5, 0.05, [])
        assert graph.delete_node(s)
        assert graph.get_edge(s, t) is None

    def test_double_delete_edge(self, graph: CausalGraph) -> None:
        s = graph.add_node("S", "primitive")
        t = graph.add_node("T", "outcome")
        graph.add_edge(s, t, 0.5, 0.05, [])
        assert graph.delete_edge(s, t)
        assert not graph.delete_edge(s, t)

    def test_multiple_edges_different_targets(self, graph: CausalGraph) -> None:
        s = graph.add_node("Source", "primitive")
        t1 = graph.add_node("T1", "outcome")
        t2 = graph.add_node("T2", "outcome")
        assert graph.add_edge(s, t1, 0.5, 0.05, [])
        assert graph.add_edge(s, t2, 0.7, 0.01, [])
        causes = graph.find_causes(t1)
        assert len(causes) == 1

    def test_constraint_enforced(self, graph: CausalGraph) -> None:
        nid1 = graph.add_node("DupName", "primitive")
        nid2 = graph.add_node("DupName", "primitive")
        assert nid1 != nid2
        nodes = graph.find_nodes_by_name("DupName")
        assert len(nodes) == 2
