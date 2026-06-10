"""Tests for SessionMemory (Redis-backed campaign state cache)."""

import os
import time
from typing import Any, Dict, Generator

import pytest

from knowledge.session_memory import SessionMemory

pytest.importorskip("testcontainers.redis")
from testcontainers.redis import RedisContainer  # type: ignore[import-untyped]

REDIS_IMAGE = os.environ.get("REDIS_IMAGE", "redis:7")


@pytest.fixture(scope="module")
def redis_url() -> Generator[str, None, None]:
    container = RedisContainer(image=REDIS_IMAGE)
    container.start()
    time.sleep(1)
    try:
        host = container.get_container_host_ip()
        port = container.get_exposed_port(6379)
        yield f"redis://{host}:{port}"
    finally:
        container.stop()


@pytest.fixture
def sm(redis_url: str) -> Generator[SessionMemory, None, None]:
    m = SessionMemory(redis_url=redis_url, ttl=3600)
    m.client.flushall()
    yield m
    m.client.flushall()
    m.close()


# ---------------------------------------------------------------------------
# Create / Get / Delete
# ---------------------------------------------------------------------------


class TestSessionCRUD:
    def test_create_session(self, sm: SessionMemory) -> None:
        assert sm.create_session("campaign_1", "GPT-4o", {"env": "test"})
        data = sm.get_session("campaign_1")
        assert data is not None
        assert data["campaign_id"] == "campaign_1"
        assert data["target_model"] == "GPT-4o"
        assert data["status"] == "running"
        assert data["metadata"] == {"env": "test"}

    def test_create_duplicate(self, sm: SessionMemory) -> None:
        assert sm.create_session("dup", "model")
        assert not sm.create_session("dup", "model")

    def test_get_session_nonexistent(self, sm: SessionMemory) -> None:
        assert sm.get_session("nonexistent") is None

    def test_delete_session(self, sm: SessionMemory) -> None:
        sm.create_session("del_me", "model")
        assert sm.delete_session("del_me")
        assert sm.get_session("del_me") is None

    def test_delete_nonexistent(self, sm: SessionMemory) -> None:
        assert not sm.delete_session("nonexistent")


class TestSessionDefaults:
    def test_default_values(self, sm: SessionMemory) -> None:
        sm.create_session("c1", "Llama-3")
        data = sm.get_session("c1")
        assert data is not None
        assert data["current_best_program_id"] == ""
        assert data["current_best_accuracy"] == 0.0
        assert data["iteration"] == 0
        assert data["intervention_count"] == 0
        assert data["hypothesis_ids"] == []

    def test_session_exists(self, sm: SessionMemory) -> None:
        sm.create_session("c1", "model")
        assert sm.session_exists("c1")
        assert not sm.session_exists("nonexistent")


# ---------------------------------------------------------------------------
# Updates
# ---------------------------------------------------------------------------


class TestSessionUpdate:
    def test_update_session(self, sm: SessionMemory) -> None:
        sm.create_session("c1", "GPT-4o")
        assert sm.update_session("c1", {"status": "completed"})
        data = sm.get_session("c1")
        assert data is not None
        assert data["status"] == "completed"

    def test_update_nonexistent(self, sm: SessionMemory) -> None:
        assert not sm.update_session("nonexistent", {"status": "completed"})

    def test_update_best_program(self, sm: SessionMemory) -> None:
        sm.create_session("c1", "GPT-4o")
        assert sm.set_best_program("c1", "prog_123", 0.95)
        data = sm.get_session("c1")
        assert data is not None
        assert data["current_best_program_id"] == "prog_123"
        assert data["current_best_accuracy"] == 0.95


# ---------------------------------------------------------------------------
# Increment operations
# ---------------------------------------------------------------------------


class TestIncrement:
    def test_increment_iteration(self, sm: SessionMemory) -> None:
        sm.create_session("c1", "model")
        assert sm.increment_iteration("c1") == 1
        assert sm.increment_iteration("c1") == 2
        data = sm.get_session("c1")
        assert data is not None
        assert data["iteration"] == 2

    def test_increment_intervention_count(self, sm: SessionMemory) -> None:
        sm.create_session("c1", "model")
        assert sm.increment_intervention_count("c1") == 1
        assert sm.increment_intervention_count("c1", 3) == 4
        data = sm.get_session("c1")
        assert data is not None
        assert data["intervention_count"] == 4


# ---------------------------------------------------------------------------
# Hypothesis list management
# ---------------------------------------------------------------------------


class TestHypotheses:
    def test_add_hypothesis(self, sm: SessionMemory) -> None:
        sm.create_session("c1", "model")
        assert sm.add_hypothesis("c1", "hyp_001")
        assert sm.add_hypothesis("c1", "hyp_002")
        hyps = sm.list_hypotheses("c1")
        assert hyps == ["hyp_001", "hyp_002"]

    def test_remove_hypothesis(self, sm: SessionMemory) -> None:
        sm.create_session("c1", "model")
        sm.add_hypothesis("c1", "hyp_001")
        sm.add_hypothesis("c1", "hyp_002")
        assert sm.remove_hypothesis("c1", "hyp_001")
        hyps = sm.list_hypotheses("c1")
        assert hyps == ["hyp_002"]

    def test_get_session_includes_hypotheses(self, sm: SessionMemory) -> None:
        sm.create_session("c1", "model")
        sm.add_hypothesis("c1", "hyp_a")
        sm.add_hypothesis("c1", "hyp_b")
        data = sm.get_session("c1")
        assert data is not None
        assert "hyp_a" in data["hypothesis_ids"]
        assert "hyp_b" in data["hypothesis_ids"]


# ---------------------------------------------------------------------------
# Status management
# ---------------------------------------------------------------------------


class TestStatus:
    def test_set_status(self, sm: SessionMemory) -> None:
        sm.create_session("c1", "model")
        assert sm.set_status("c1", "completed")
        data = sm.get_session("c1")
        assert data is not None
        assert data["status"] == "completed"

    def test_set_status_nonexistent(self, sm: SessionMemory) -> None:
        assert not sm.set_status("nonexistent", "completed")


# ---------------------------------------------------------------------------
# Session listing
# ---------------------------------------------------------------------------


class TestListSessions:
    def test_list_active_sessions(self, sm: SessionMemory) -> None:
        sm.create_session("c1", "model_a")
        sm.create_session("c2", "model_b")
        sessions = sm.list_active_sessions()
        assert "c1" in sessions
        assert "c2" in sessions

    def test_list_active_sessions_empty(self, sm: SessionMemory) -> None:
        assert sm.list_active_sessions() == []


# ---------------------------------------------------------------------------
# Context manager
# ---------------------------------------------------------------------------


class TestContextManager:
    def test_context_manager(self, redis_url: str) -> None:
        with SessionMemory(redis_url=redis_url) as sm:
            sm.create_session("ctx_test", "model")
            assert sm.session_exists("ctx_test")
