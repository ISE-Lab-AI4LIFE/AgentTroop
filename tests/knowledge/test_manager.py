"""Tests for KnowledgeManager (Redis + in-memory proposal queue)."""

import os
import time
from typing import Any, Dict, Generator, List
from unittest.mock import MagicMock

import pytest

from knowledge.manager import (
    DEFAULT_OWNERS,
    KnowledgeManager,
    Proposal,
    Target,
)

pytest.importorskip("testcontainers.redis")
from testcontainers.redis import RedisContainer  # type: ignore[import-untyped]

REDIS_IMAGE = os.environ.get("REDIS_IMAGE", "redis:7")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def redis_url() -> Generator[str, None, None]:
    container = RedisContainer(image=REDIS_IMAGE)
    container.start()
    time.sleep(1)
    try:
        yield container.get_connection_url()
    finally:
        container.stop()


@pytest.fixture
def mgr_memory() -> Generator[KnowledgeManager, None, None]:
    mgr = KnowledgeManager(use_redis=False)
    _register_mock_stores(mgr)
    yield mgr
    mgr.close()


@pytest.fixture
def mgr_redis(redis_url: str) -> Generator[KnowledgeManager, None, None]:
    mgr = KnowledgeManager(redis_url=redis_url, use_redis=True)
    _register_mock_stores(mgr)
    yield mgr
    mgr.close()


def _register_mock_stores(mgr: KnowledgeManager) -> None:
    for target in Target:
        store = MagicMock()
        store.save.return_value = "saved_ok"
        store.add.return_value = "added_ok"
        store.get.return_value = {"data": "mock"}
        store.find.return_value = [{"data": "mock"}]
        mgr.register_store(target.value, store)


# ---------------------------------------------------------------------------
# Target enum & default owners
# ---------------------------------------------------------------------------


class TestTargetEnum:
    def test_all_targets_have_owners(self) -> None:
        for t in Target:
            assert t.value in DEFAULT_OWNERS, f"{t.value} has no default owners"

    def test_owners_non_empty(self) -> None:
        for t, owners in DEFAULT_OWNERS.items():
            assert len(owners) > 0, f"{t} has empty owners list"


# ---------------------------------------------------------------------------
# Proposal dataclass
# ---------------------------------------------------------------------------


class TestProposal:
    def test_auto_generates_id(self) -> None:
        p = Proposal(target="episodic", data="x", agent_id="AgentA")
        assert p.proposal_id.startswith("prp_")

    def test_default_status(self) -> None:
        p = Proposal(target="episodic", data="x", agent_id="AgentA")
        assert p.status == "pending"

    def test_to_dict_roundtrip(self) -> None:
        p1 = Proposal(
            target="causal_graph",
            data={"key": "val"},
            agent_id="CognitiveAgent",
            action="update",
        )
        data = p1.to_dict()
        p2 = Proposal.from_dict(data)
        for attr in ("proposal_id", "target", "action", "agent_id", "status"):
            assert getattr(p1, attr) == getattr(p2, attr)

    def test_from_dict_full(self) -> None:
        data = {
            "proposal_id": "prp_test",
            "target": "scientific_memory",
            "action": "create",
            "data": {"theory": "test"},
            "agent_id": "ResearcherAgent",
            "timestamp": 1000.0,
            "status": "accepted",
            "result": "ok",
            "error": None,
        }
        p = Proposal.from_dict(data)
        assert p.proposal_id == "prp_test"
        assert p.status == "accepted"


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------


class TestInit:
    def test_memory_backend(self) -> None:
        mgr = KnowledgeManager(use_redis=False)
        assert mgr._redis_client is None
        assert len(mgr._memory_queues) == len(Target)
        mgr.close()

    def test_redis_backend_fallback_to_memory(self) -> None:
        mgr = KnowledgeManager(redis_url="redis://localhost:9999", use_redis=True)
        assert mgr._redis_client is None
        assert len(mgr._memory_queues) == len(Target)
        mgr.close()

    def test_context_manager(self) -> None:
        with KnowledgeManager(use_redis=False) as mgr:
            assert mgr is not None

    def test_register_store_invalid_target(self) -> None:
        mgr = KnowledgeManager(use_redis=False)
        with pytest.raises(ValueError, match="Unknown target"):
            mgr.register_store("invalid", MagicMock())
        mgr.close()


# ---------------------------------------------------------------------------
# Read / Write — permission
# ---------------------------------------------------------------------------


class TestRead:
    def test_read_delegates_to_store(self, mgr_memory: KnowledgeManager) -> None:
        result = mgr_memory.read(Target.EPISODIC.value, "query")
        store = mgr_memory.get_store(Target.EPISODIC.value)
        store.get.assert_called_with("query")

    def test_read_unregistered_target(self, mgr_memory: KnowledgeManager) -> None:
        mgr_unreg = KnowledgeManager(use_redis=False)
        with pytest.raises(ValueError, match="No store registered"):
            mgr_unreg.read(Target.EPISODIC.value, "q")
        mgr_unreg.close()


class TestWritePermission:
    def test_owner_can_write(self, mgr_memory: KnowledgeManager) -> None:
        result = mgr_memory.write(
            Target.DEFENSE_STORE.value,
            {"program": "test"},
            "ResearcherAgent",
        )
        assert result is not None

    def test_non_owner_cannot_write(self, mgr_memory: KnowledgeManager) -> None:
        with pytest.raises(PermissionError, match="not an owner"):
            mgr_memory.write(
                Target.DEFENSE_STORE.value,
                {"program": "test"},
                "CognitiveAgent",
            )

    def test_owner_can_write_episodic(self, mgr_memory: KnowledgeManager) -> None:
        result = mgr_memory.write(
            Target.EPISODIC.value, {"episode": "data"}, "CognitiveAgent",
        )
        assert result is not None

    def test_non_owner_cannot_write_causal(
        self, mgr_memory: KnowledgeManager
    ) -> None:
        with pytest.raises(PermissionError):
            mgr_memory.write(
                Target.CAUSAL.value, {"node": "x"}, "StrategistAgent",
            )

    def test_owners_list(self, mgr_memory: KnowledgeManager) -> None:
        owners = mgr_memory.get_owners(Target.SCIENTIFIC.value)
        assert "ResearcherAgent" in owners

    def test_is_owner(self, mgr_memory: KnowledgeManager) -> None:
        assert mgr_memory.is_owner(Target.CAUSAL.value, "ResearcherAgent")
        assert not mgr_memory.is_owner(Target.CAUSAL.value, "CognitiveAgent")

    def test_set_owners(self, mgr_memory: KnowledgeManager) -> None:
        mgr_memory.set_owners(Target.CAUSAL.value, ["NewAgent"])
        assert mgr_memory.is_owner(Target.CAUSAL.value, "NewAgent")
        assert not mgr_memory.is_owner(Target.CAUSAL.value, "ResearcherAgent")


# ---------------------------------------------------------------------------
# Proposal flow — in-memory backend
# ---------------------------------------------------------------------------


class TestProposalFlowMemory:
    def test_propose_returns_id(self, mgr_memory: KnowledgeManager) -> None:
        pid = mgr_memory.propose(
            Target.CAUSAL.value,
            {"edge": "test"},
            "CognitiveAgent",
        )
        assert pid.startswith("prp_")

    def test_propose_invalid_target(self, mgr_memory: KnowledgeManager) -> None:
        with pytest.raises(ValueError, match="Unknown target"):
            mgr_memory.propose("invalid", {}, "Agent")

    def test_poll_returns_proposals(self, mgr_memory: KnowledgeManager) -> None:
        pid = mgr_memory.propose(
            Target.CAUSAL.value, {"data": "x"}, "CognitiveAgent",
        )
        proposals = mgr_memory.poll_proposals("ResearcherAgent", Target.CAUSAL.value)
        assert len(proposals) >= 1
        assert proposals[0]["proposal_id"] == pid

    def test_poll_non_owner_returns_empty(
        self, mgr_memory: KnowledgeManager
    ) -> None:
        mgr_memory.propose(
            Target.CAUSAL.value, {"data": "x"}, "CognitiveAgent",
        )
        proposals = mgr_memory.poll_proposals(
            "CognitiveAgent", Target.CAUSAL.value
        )
        assert proposals == []

    def test_poll_empty_queue(self, mgr_memory: KnowledgeManager) -> None:
        proposals = mgr_memory.poll_proposals(
            "ResearcherAgent", Target.SCIENTIFIC.value
        )
        assert proposals == []

    def test_resolve_accepted(self, mgr_memory: KnowledgeManager) -> None:
        pid = mgr_memory.propose(
            Target.CAUSAL.value, {}, "CognitiveAgent",
        )
        mgr_memory.poll_proposals("ResearcherAgent", Target.CAUSAL.value)
        assert mgr_memory.resolve_proposal(pid, accepted=True, result="ok")
        status = mgr_memory.get_proposal_status(pid)
        assert status is not None
        assert status["status"] == "accepted"

    def test_resolve_rejected(self, mgr_memory: KnowledgeManager) -> None:
        pid = mgr_memory.propose(
            Target.CAUSAL.value, {}, "CognitiveAgent",
        )
        mgr_memory.poll_proposals("ResearcherAgent", Target.CAUSAL.value)
        assert mgr_memory.resolve_proposal(
            pid, accepted=False, error="Invalid data"
        )
        status = mgr_memory.get_proposal_status(pid)
        assert status is not None
        assert status["status"] == "rejected"
        assert status["error"] == "Invalid data"

    def test_resolve_nonexistent(self, mgr_memory: KnowledgeManager) -> None:
        assert not mgr_memory.resolve_proposal("nonexistent", True)

    def test_get_proposal_status_nonexistent(
        self, mgr_memory: KnowledgeManager
    ) -> None:
        assert mgr_memory.get_proposal_status("nonexistent") is None

    def test_propose_multiple_poll_all(
        self, mgr_memory: KnowledgeManager
    ) -> None:
        pids = []
        for i in range(5):
            pid = mgr_memory.propose(
                Target.CAUSAL.value, {"i": i}, "CognitiveAgent",
            )
            pids.append(pid)

        proposals = mgr_memory.poll_proposals(
            "ResearcherAgent", Target.CAUSAL.value
        )
        assert len(proposals) == 5

    def test_full_flow(self, mgr_memory: KnowledgeManager) -> None:
        pid = mgr_memory.propose(
            Target.CAUSAL.value,
            {"node": "ROT13", "type": "transform"},
            "CognitiveAgent",
            action="create",
        )
        proposals = mgr_memory.poll_proposals(
            "ResearcherAgent", Target.CAUSAL.value
        )
        assert len(proposals) == 1
        assert proposals[0]["action"] == "create"
        assert proposals[0]["agent_id"] == "CognitiveAgent"

        mgr_memory.resolve_proposal(pid, accepted=True, result="node_created")
        status = mgr_memory.get_proposal_status(pid)
        assert status["status"] == "accepted"
        assert status["result"] == "node_created"


# ---------------------------------------------------------------------------
# Proposal flow — Redis backend
# ---------------------------------------------------------------------------


class TestProposalFlowRedis:
    def test_redis_propose_and_poll(
        self, mgr_redis: KnowledgeManager
    ) -> None:
        pid = mgr_redis.propose(
            Target.CAUSAL.value,
            {"edge": "test"},
            "CognitiveAgent",
        )
        assert mgr_redis._redis_client is not None
        proposals = mgr_redis.poll_proposals(
            "ResearcherAgent", Target.CAUSAL.value
        )
        assert len(proposals) >= 1
        assert proposals[0]["proposal_id"] == pid

    def test_redis_resolve(self, mgr_redis: KnowledgeManager) -> None:
        pid = mgr_redis.propose(
            Target.CAUSAL.value, {"data": "x"}, "CognitiveAgent",
        )
        mgr_redis.poll_proposals("ResearcherAgent", Target.CAUSAL.value)
        assert mgr_redis.resolve_proposal(pid, accepted=True, result="done")
        status = mgr_redis.get_proposal_status(pid)
        assert status is not None
        assert status["status"] == "accepted"

    def test_redis_reject(self, mgr_redis: KnowledgeManager) -> None:
        pid = mgr_redis.propose(
            Target.CAUSAL.value, {}, "CognitiveAgent",
        )
        mgr_redis.poll_proposals("ResearcherAgent", Target.CAUSAL.value)
        assert mgr_redis.resolve_proposal(
            pid, accepted=False, error="invalid"
        )
        status = mgr_redis.get_proposal_status(pid)
        assert status["status"] == "rejected"
        assert status["error"] == "invalid"

    def test_redis_multiple_proposals(
        self, mgr_redis: KnowledgeManager
    ) -> None:
        for i in range(3):
            mgr_redis.propose(
                Target.CAUSAL.value,
                {"i": i}, "CognitiveAgent",
            )
        proposals = mgr_redis.poll_proposals(
            "ResearcherAgent", Target.CAUSAL.value
        )
        assert len(proposals) == 3

    def test_redis_empty_poll(self, mgr_redis: KnowledgeManager) -> None:
        proposals = mgr_redis.poll_proposals(
            "ResearcherAgent", Target.CAUSAL.value
        )
        assert proposals == []

    def test_redis_non_owner_poll(self, mgr_redis: KnowledgeManager) -> None:
        mgr_redis.propose(
            Target.CAUSAL.value, {}, "CognitiveAgent",
        )
        proposals = mgr_redis.poll_proposals(
            "CognitiveAgent", Target.CAUSAL.value
        )
        assert proposals == []

    def test_redis_get_status_nonexistent(
        self, mgr_redis: KnowledgeManager
    ) -> None:
        assert mgr_redis.get_proposal_status("nonexistent") is None


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_propose_before_store_registered(
        self, mgr_memory: KnowledgeManager
    ) -> None:
        pid = mgr_memory.propose(
            Target.CAUSAL.value, {}, "CognitiveAgent",
        )
        assert pid.startswith("prp_")

    def test_multiple_targets_independent(
        self, mgr_memory: KnowledgeManager
    ) -> None:
        pid1 = mgr_memory.propose(
            Target.CAUSAL.value, {}, "CognitiveAgent",
        )
        pid2 = mgr_memory.propose(
            Target.SCIENTIFIC.value, {}, "StrategistAgent",
        )
        cau_proposals = mgr_memory.poll_proposals(
            "ResearcherAgent", Target.CAUSAL.value
        )
        sci_proposals = mgr_memory.poll_proposals(
            "ResearcherAgent", Target.SCIENTIFIC.value
        )
        assert len(cau_proposals) == 1
        assert len(sci_proposals) == 1
        assert cau_proposals[0]["proposal_id"] == pid1
        assert sci_proposals[0]["proposal_id"] == pid2

    def test_write_different_targets(
        self, mgr_memory: KnowledgeManager
    ) -> None:
        result = mgr_memory.write(
            Target.EPISODIC.value, {"e": 1}, "CognitiveAgent",
        )
        assert result is not None

        with pytest.raises(PermissionError):
            mgr_memory.write(
                Target.DEFENSE_STORE.value, {}, "CognitiveAgent",
            )
