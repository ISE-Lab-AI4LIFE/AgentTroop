"""Tests for Scientific Memory (Neo4j-backed, versioned, dynamic conditions)."""

import json
import os
import tempfile
import time
from typing import Any, Dict

import pytest

from knowledge.scientific_memory import ScientificMemory, Theory

# ---------------------------------------------------------------------------
# Connectivity check
# ---------------------------------------------------------------------------

NEO4J_URI = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.environ.get("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD", "password")


def _neo4j_reachable() -> bool:
    try:
        m = ScientificMemory(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
        m._ensure_constraints()
        m.close()
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def memory() -> ScientificMemory:
    m = ScientificMemory(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
    m._ensure_constraints()
    m.delete_all()
    yield m
    m.delete_all()
    m.close()


def make_theory(**overrides: Any) -> Theory:
    base: Dict[str, Any] = dict(
        pattern="IF contains(x, 'kill') THEN REFUSE",
        conditions={"model_family": "default"},
        confidence=0.75,
        provenance=["ep_001"],
    )
    base.update(overrides)
    return Theory(**base)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

neo4j_required = pytest.mark.skipif(
    not _neo4j_reachable(),
    reason="Neo4j is not reachable — start it with: "
    "docker run --rm -p 7687:7687 -e NEO4J_AUTH=neo4j/password neo4j:5",
)


# ---------------------------------------------------------------------------
# Theory dataclass unit tests (no Neo4j needed)
# ---------------------------------------------------------------------------


class TestTheoryDataclass:
    def test_auto_generates_id(self) -> None:
        t = Theory(pattern="test")
        assert t.id.startswith("thr_")
        assert len(t.id) > 4

    def test_clamps_confidence(self) -> None:
        t = Theory(pattern="x", confidence=1.5)
        assert t.confidence == 1.0
        t2 = Theory(pattern="x", confidence=-0.5)
        assert t2.confidence == 0.0

    def test_default_version(self) -> None:
        t = Theory(pattern="x")
        assert t.version == 1

    def test_clamps_version(self) -> None:
        t = Theory(pattern="x", version=0)
        assert t.version == 1
        t2 = Theory(pattern="x", version=-5)
        assert t2.version == 1

    def test_to_dict_roundtrip(self) -> None:
        t1 = Theory(
            pattern="IF p(x) THEN REFUSE",
            conditions={"cipher": "base64"},
            confidence=0.88,
            provenance=["ep_a", "ep_b"],
            version=3,
            metadata={"source": "manual"},
        )
        data = t1.to_dict()
        t2 = Theory.from_dict(data)
        for attr in (
            "id", "pattern", "confidence", "provenance",
            "version", "created_at", "updated_at",
        ):
            assert getattr(t1, attr) == getattr(t2, attr), f"Mismatch on {attr}"
        assert t2.conditions == {"cipher": "base64"}
        assert t2.metadata == {"source": "manual"}


# ---------------------------------------------------------------------------
# Integration tests (require Neo4j)
# ---------------------------------------------------------------------------


@neo4j_required
class TestScientificMemory:
    """Scientific Memory integration tests — skipped if Neo4j unavailable."""

    # ------------------------------------------------------------------
    # CRUD — versioning
    # ------------------------------------------------------------------

    def test_save_creates_version_1(self, memory: ScientificMemory) -> None:
        t = make_theory()
        tid = memory.save_theory(t)
        retrieved = memory.get_theory(tid)
        assert retrieved is not None
        assert retrieved.version == 1

    def test_save_increments_version(self, memory: ScientificMemory) -> None:
        t = make_theory(id="ver_inc_test", pattern="v1")
        memory.save_theory(t)
        memory.save_theory(make_theory(id="ver_inc_test", pattern="v2"))
        retrieved = memory.get_theory("ver_inc_test")
        assert retrieved is not None
        assert retrieved.version == 2
        assert retrieved.pattern == "v2"

    def test_get_theory_version(self, memory: ScientificMemory) -> None:
        t = make_theory(id="get_ver_test", pattern="first", confidence=0.3)
        memory.save_theory(t)
        memory.save_theory(
            make_theory(id="get_ver_test", pattern="second", confidence=0.9)
        )
        v1 = memory.get_theory_version("get_ver_test", 1)
        assert v1 is not None
        assert v1.pattern == "first"
        assert v1.confidence == 0.3
        assert v1.version == 1

    def test_get_theory_version_nonexistent(
        self, memory: ScientificMemory
    ) -> None:
        assert memory.get_theory_version("no_such_id", 1) is None
        t = make_theory()
        memory.save_theory(t)
        assert memory.get_theory_version(t.id, 99) is None

    def test_get_nonexistent_returns_none(self, memory: ScientificMemory) -> None:
        assert memory.get_theory("nonexistent_id") is None

    def test_save_preserves_created_at(self, memory: ScientificMemory) -> None:
        t = make_theory(id="preserve_ca", pattern="v1")
        memory.save_theory(t)
        original_ts = t.created_at
        time.sleep(0.01)
        memory.save_theory(make_theory(id="preserve_ca", pattern="v2"))
        retrieved = memory.get_theory("preserve_ca")
        assert retrieved is not None
        assert retrieved.created_at == original_ts
        assert retrieved.updated_at > original_ts

    # ------------------------------------------------------------------
    # Dynamic conditions filtering (Cypher-level)
    # ------------------------------------------------------------------

    def test_find_by_conditions(self, memory: ScientificMemory) -> None:
        memory.delete_all()
        memory.save_theory(
            make_theory(pattern="rlhf", conditions={"model_family": "RLHF"})
        )
        memory.save_theory(
            make_theory(pattern="cai", conditions={"model_family": "CAI"})
        )
        results = memory.find_theories(conditions={"model_family": "RLHF"})
        assert len(results) == 1
        assert results[0].pattern == "rlhf"

    def test_find_by_conditions_no_match(self, memory: ScientificMemory) -> None:
        memory.delete_all()
        memory.save_theory(
            make_theory(pattern="x", conditions={"family": "A"})
        )
        results = memory.find_theories(conditions={"family": "B"})
        assert len(results) == 0

    def test_find_by_min_confidence(self, memory: ScientificMemory) -> None:
        memory.delete_all()
        memory.save_theory(make_theory(confidence=0.4))
        memory.save_theory(make_theory(confidence=0.7))
        memory.save_theory(make_theory(confidence=0.9))
        results = memory.find_theories(min_confidence=0.7)
        assert len(results) == 2

    def test_find_by_conditions_and_confidence(
        self, memory: ScientificMemory
    ) -> None:
        memory.delete_all()
        memory.save_theory(
            make_theory(
                pattern="good",
                conditions={"family": "A"},
                confidence=0.95,
            )
        )
        memory.save_theory(
            make_theory(
                pattern="weak",
                conditions={"family": "A"},
                confidence=0.3,
            )
        )
        results = memory.find_theories(
            conditions={"family": "A"}, min_confidence=0.8
        )
        assert len(results) == 1
        assert results[0].pattern == "good"

    def test_find_by_multiple_conditions(self, memory: ScientificMemory) -> None:
        memory.delete_all()
        memory.save_theory(
            make_theory(
                pattern="rot13",
                conditions={"family": "RLHF", "cipher": "rot13"},
            )
        )
        memory.save_theory(
            make_theory(
                pattern="base64",
                conditions={"family": "RLHF", "cipher": "base64"},
            )
        )
        results = memory.find_theories(
            conditions={"family": "RLHF", "cipher": "rot13"}
        )
        assert len(results) == 1
        assert results[0].pattern == "rot13"

    def test_empty_conditions_in_find(self, memory: ScientificMemory) -> None:
        results = memory.find_theories(conditions={})
        all_results = memory.find_theories()
        assert len(results) == len(all_results)

    def test_zero_confidence_filter(self, memory: ScientificMemory) -> None:
        memory.save_theory(make_theory(confidence=0.0))
        memory.save_theory(make_theory(confidence=0.5))
        results = memory.find_theories(min_confidence=0.0)
        assert len(results) >= 2

    # ------------------------------------------------------------------
    # find_theories_by_pattern
    # ------------------------------------------------------------------

    def test_find_by_pattern_substring(self, memory: ScientificMemory) -> None:
        memory.delete_all()
        memory.save_theory(
            make_theory(pattern="IF contains(decode_rot13(x), 'bomb')")
        )
        memory.save_theory(
            make_theory(pattern="IF contains(x, 'kill') THEN REFUSE")
        )
        results = memory.find_theories_by_pattern("rot13")
        assert len(results) == 1
        assert "rot13" in results[0].pattern

    def test_find_by_pattern_case_insensitive(
        self, memory: ScientificMemory
    ) -> None:
        memory.delete_all()
        memory.save_theory(
            make_theory(pattern="REFUSE when prompt has BOMB")
        )
        results = memory.find_theories_by_pattern("bomb", case_sensitive=False)
        assert len(results) == 1

    def test_find_by_pattern_case_sensitive(self, memory: ScientificMemory) -> None:
        memory.delete_all()
        memory.save_theory(
            make_theory(pattern="REFUSE when prompt has bomb")
        )
        results = memory.find_theories_by_pattern("BOMB", case_sensitive=True)
        assert len(results) == 0

    def test_find_by_pattern_with_confidence(
        self, memory: ScientificMemory
    ) -> None:
        memory.delete_all()
        memory.save_theory(
            make_theory(pattern="rule with rot13", confidence=0.3)
        )
        memory.save_theory(
            make_theory(pattern="another rot13 rule", confidence=0.9)
        )
        results = memory.find_theories_by_pattern(
            "rot13", min_confidence=0.8
        )
        assert len(results) == 1
        assert results[0].confidence == 0.9

    # ------------------------------------------------------------------
    # Update confidence (creates new version)
    # ------------------------------------------------------------------

    def test_update_confidence(self, memory: ScientificMemory) -> None:
        t = make_theory(confidence=0.3)
        memory.save_theory(t)
        ok = memory.update_confidence(t.id, 0.95)
        assert ok is True
        retrieved = memory.get_theory(t.id)
        assert retrieved is not None
        assert retrieved.confidence == 0.95
        assert retrieved.version == 2

    def test_update_confidence_nonexistent(self, memory: ScientificMemory) -> None:
        ok = memory.update_confidence("no_such_id", 0.5)
        assert ok is False

    def test_update_confidence_clamps(self, memory: ScientificMemory) -> None:
        t = make_theory()
        memory.save_theory(t)
        memory.update_confidence(t.id, 2.0)
        retrieved = memory.get_theory(t.id)
        assert retrieved is not None
        assert retrieved.confidence == 1.0

    # ------------------------------------------------------------------
    # Delete
    # ------------------------------------------------------------------

    def test_delete_theory_removes_all_versions(
        self, memory: ScientificMemory
    ) -> None:
        t = make_theory(id="del_all_ver")
        memory.save_theory(t)
        memory.save_theory(make_theory(id="del_all_ver", pattern="v2"))
        assert memory.get_theory("del_all_ver") is not None
        ok = memory.delete_theory("del_all_ver")
        assert ok is True
        assert memory.get_theory("del_all_ver") is None
        assert memory.get_theory_version("del_all_ver", 1) is None
        assert memory.get_theory_version("del_all_ver", 2) is None

    def test_delete_nonexistent(self, memory: ScientificMemory) -> None:
        ok = memory.delete_theory("no_such_id")
        assert ok is False

    def test_delete_all(self, memory: ScientificMemory) -> None:
        memory.save_theory(make_theory())
        memory.save_theory(make_theory())
        memory.save_theory(make_theory())
        count = memory.delete_all()
        assert count >= 3
        assert len(memory.find_theories()) == 0

    # ------------------------------------------------------------------
    # Auto-compact
    # ------------------------------------------------------------------

    def _create_versions(
        self, mem: ScientificMemory, tid: str, n: int
    ) -> None:
        for i in range(1, n + 1):
            mem.save_theory(
                make_theory(id=tid, pattern=f"v{i}", confidence=i / n)
            )

    def test_compact_keeps_latest_versions(
        self, memory: ScientificMemory
    ) -> None:
        self._create_versions(memory, "compact_keep", 20)
        deleted = memory.compact_theory("compact_keep", keep_versions=5)
        assert deleted == 15
        v = memory.get_theory("compact_keep")
        assert v is not None
        assert v.version == 20
        v16 = memory.get_theory_version("compact_keep", 16)
        assert v16 is not None
        v15 = memory.get_theory_version("compact_keep", 15)
        assert v15 is None

    def test_compact_nonexistent_returns_zero(
        self, memory: ScientificMemory
    ) -> None:
        deleted = memory.compact_theory("no_such_id")
        assert deleted == 0

    def test_compact_enforces_minimum_one(
        self, memory: ScientificMemory
    ) -> None:
        self._create_versions(memory, "compact_min1", 5)
        deleted = memory.compact_theory("compact_min1", keep_versions=0)
        assert deleted == 4
        assert memory.get_theory("compact_min1") is not None

    def test_compact_when_below_threshold(
        self, memory: ScientificMemory
    ) -> None:
        self._create_versions(memory, "compact_below", 3)
        deleted = memory.compact_theory("compact_below", keep_versions=10)
        assert deleted == 0

    def test_compact_all(self, memory: ScientificMemory) -> None:
        memory.delete_all()
        self._create_versions(memory, "ca_a", 15)
        self._create_versions(memory, "ca_b", 12)
        self._create_versions(memory, "ca_c", 3)
        summary = memory.compact_all(keep_versions=10)
        assert len(summary) == 2  # ca_a (5 deleted) + ca_b (2 deleted)
        assert summary.get("ca_a") == 5
        assert summary.get("ca_b") == 2
        assert "ca_c" not in summary
        assert memory.get_theory_version("ca_a", 5) is None
        assert memory.get_theory_version("ca_a", 15) is not None

    # ------------------------------------------------------------------
    # Advanced auto-compact
    # ------------------------------------------------------------------

    def test_compact_if_needed_triggers_when_over_threshold(
        self, memory: ScientificMemory
    ) -> None:
        memory.delete_all()
        self._create_versions(memory, "trigger_t", 25)
        summary = memory.compact_if_needed(
            keep_versions=10, max_versions_before_compact=20
        )
        assert "trigger_t" in summary
        assert summary["trigger_t"] == 15  # 25 - 10
        retrieved = memory.get_theory("trigger_t")
        assert retrieved is not None
        assert retrieved.version == 25

    def test_compact_if_needed_noop_when_under_threshold(
        self, memory: ScientificMemory
    ) -> None:
        memory.delete_all()
        self._create_versions(memory, "under_t", 5)
        summary = memory.compact_if_needed(
            keep_versions=10, max_versions_before_compact=20
        )
        assert "under_t" not in summary or summary["under_t"] == 0

    def test_compact_if_needed_multiple_theories(
        self, memory: ScientificMemory
    ) -> None:
        memory.delete_all()
        self._create_versions(memory, "multi_a", 30)
        self._create_versions(memory, "multi_b", 10)
        self._create_versions(memory, "multi_c", 25)
        summary = memory.compact_if_needed(
            keep_versions=10, max_versions_before_compact=20
        )
        assert "multi_a" in summary  # 30 > 20 → compacted to 10 → 20 deleted
        assert "multi_b" not in summary  # 10 < 20
        assert "multi_c" in summary  # 25 > 20

    def _backdate_theory(
        self, memory: ScientificMemory, tid: str, version: int, days_ago: float
    ) -> None:
        """Set a theory version's timestamps to the past via raw Cypher."""
        ts = time.time() - days_ago * 86400
        with memory._session() as session:
            session.run(
                """MATCH (t:Theory {id: $id, version: $ver})
                   SET t.created_at = $ts, t.updated_at = $ts
                """,
                id=tid, ver=version, ts=ts,
            )

    def test_compact_older_than(self, memory: ScientificMemory) -> None:
        memory.delete_all()
        tid = "older_test"
        for i in range(1, 6):
            memory.save_theory(make_theory(id=tid, pattern=f"old_v{i}", confidence=0.5))
            self._backdate_theory(memory, tid, i, 30 + i)  # 31–35 days ago

        # compact older than 20 days, keep 1
        summary = memory.compact_older_than(days=20, keep_versions=1)
        assert tid in summary
        assert summary[tid] == 4  # 4 old versions deleted, 1 kept

    def test_compact_older_than_keeps_minimum(
        self, memory: ScientificMemory
    ) -> None:
        memory.delete_all()
        tid = "min_keep"
        for i in range(1, 4):
            memory.save_theory(make_theory(id=tid, pattern=f"min_v{i}", confidence=0.5))
            self._backdate_theory(memory, tid, i, 10 + i)  # 11–13 days ago

        # compact older than 5 days, keep 2
        summary = memory.compact_older_than(days=5, keep_versions=2)
        assert tid in summary
        assert summary[tid] == 1  # 3 total, keep 2 → 1 deleted
        v2 = memory.get_theory_version(tid, 2)
        v3 = memory.get_theory_version(tid, 3)
        assert v2 is not None
        assert v3 is not None

    def test_get_version_stats(self, memory: ScientificMemory) -> None:
        memory.delete_all()
        tid = "stats_test"
        self._create_versions(memory, tid, 7)
        stats = memory.get_version_stats(tid)
        assert stats["total_versions"] == 7
        assert stats["oldest_version"] == 1
        assert stats["newest_version"] == 7
        assert stats["oldest_updated_at"] > 0
        assert stats["newest_updated_at"] > stats["oldest_updated_at"]
        assert stats["estimated_size_bytes"] > 0

    def test_get_version_stats_nonexistent(self, memory: ScientificMemory) -> None:
        stats = memory.get_version_stats("no_such_theory")
        assert stats["total_versions"] == 0
        assert stats["oldest_version"] is None

    def test_auto_compact_enable_disable(self, memory: ScientificMemory) -> None:
        # Just test the flag toggle — the background thread is tested via logic
        assert not memory._auto_compact_enabled
        memory.set_auto_compact_enabled(True, keep_versions=5, check_interval_minutes=1)
        assert memory._auto_compact_enabled
        assert memory._auto_compact_keep_versions == 5
        memory.disable_auto_compact()
        assert not memory._auto_compact_enabled

    # ------------------------------------------------------------------
    # Export / Import
    # ------------------------------------------------------------------

    def test_export_and_import_roundtrip_latest(
        self, memory: ScientificMemory
    ) -> None:
        memory.delete_all()
        t1 = make_theory(pattern="p_a", confidence=0.9)
        t2 = make_theory(pattern="p_b", confidence=0.8)
        memory.save_theory(t1)
        memory.save_theory(t2)
        with tempfile.NamedTemporaryFile(
            suffix=".json", mode="w", delete=False
        ) as f:
            path = f.name
        try:
            memory.export_theories(path, include_history=False)
            memory.delete_all()
            count = memory.import_theories(
                path, include_history=False
            )
            assert count == 2
            patterns = {r.pattern for r in memory.find_theories()}
            assert patterns == {"p_a", "p_b"}
        finally:
            os.unlink(path)

    def test_export_and_import_with_history(
        self, memory: ScientificMemory
    ) -> None:
        memory.delete_all()
        t = make_theory(id="hist_test", pattern="v1", confidence=0.3)
        memory.save_theory(t)
        memory.save_theory(
            make_theory(id="hist_test", pattern="v2", confidence=0.9)
        )
        with tempfile.NamedTemporaryFile(
            suffix=".json", mode="w", delete=False
        ) as f:
            path = f.name
        try:
            memory.export_theories(path, include_history=True)
            memory.delete_all()
            memory.import_theories(path, include_history=True)
            # Both versions should exist
            v1 = memory.get_theory_version("hist_test", 1)
            v2 = memory.get_theory_version("hist_test", 2)
            assert v1 is not None and v1.pattern == "v1"
            assert v2 is not None and v2.pattern == "v2"
            # get_theory should return latest
            assert memory.get_theory("hist_test").pattern == "v2"
        finally:
            os.unlink(path)

    def test_import_skips_existing_version(
        self, memory: ScientificMemory
    ) -> None:
        memory.delete_all()
        t = make_theory(id="skip_dup", pattern="original")
        memory.save_theory(t)
        data = [t.to_dict()]
        with tempfile.NamedTemporaryFile(
            suffix=".json", mode="w", delete=False
        ) as f:
            json.dump(data, f)
            path = f.name
        try:
            # Import with overwrite_existing=False — should skip
            count = memory.import_theories(
                path, overwrite_existing=False
            )
            assert count == 0
        finally:
            os.unlink(path)

    def test_import_overwrites_existing(self, memory: ScientificMemory) -> None:
        memory.delete_all()
        t = make_theory(id="overwrite_dup", pattern="original")
        memory.save_theory(t)
        new_t = make_theory(
            id="overwrite_dup", pattern="overwritten"
        )
        data = [new_t.to_dict()]
        with tempfile.NamedTemporaryFile(
            suffix=".json", mode="w", delete=False
        ) as f:
            json.dump(data, f)
            path = f.name
        try:
            count = memory.import_theories(
                path, overwrite_existing=True
            )
            assert count == 1
            retrieved = memory.get_theory("overwrite_dup")
            assert retrieved is not None
            assert retrieved.pattern == "overwritten"
        finally:
            os.unlink(path)

    def test_import_handles_empty_list(self, memory: ScientificMemory) -> None:
        with tempfile.NamedTemporaryFile(
            suffix=".json", mode="w", delete=False
        ) as f:
            json.dump([], f)
            path = f.name
        try:
            count = memory.import_theories(path)
            assert count == 0
        finally:
            os.unlink(path)

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def test_context_manager(self) -> None:
        with ScientificMemory(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD) as m:
            m._ensure_constraints()
            t = make_theory()
            m.save_theory(t)
            assert m.get_theory(t.id) is not None
