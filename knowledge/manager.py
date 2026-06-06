"""Knowledge Manager — unified read/write API with single-writer ownership
and proposal queue for cross-agent writes.

Architecture:
    Each knowledge *target* (episodic, defense_store, scientific, causal_graph,
    session) has a set of *owner* agents that may write directly.  Agents that
    are **not** owners must submit a *proposal* via ``propose()``.  The owner
    polls proposals via ``poll_proposals()``, validates, writes, and resolves.

Backends:
    - Redis (production):  proposals are stored in a Redis list per target,
      proposal status is kept in a Redis hash keyed by ``proposal:<id>``.
    - In-memory (dev/test):  uses the standard library ``queue.Queue`` per
      target plus a dictionary for proposal status.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from queue import Queue as MemoryQueue
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enums & constants
# ---------------------------------------------------------------------------


class Target(str, Enum):
    EPISODIC = "episodic"
    DEFENSE_STORE = "defense_program_store"
    SCIENTIFIC = "scientific_memory"
    CAUSAL = "causal_graph"
    SESSION = "session_memory"


# Map each target to the list of agent IDs that may write directly.
DEFAULT_OWNERS: Dict[str, List[str]] = {
    Target.EPISODIC.value: ["CognitiveAgent", "StrategistAgent"],
    Target.DEFENSE_STORE.value: ["ResearcherAgent"],
    Target.SCIENTIFIC.value: ["ResearcherAgent"],
    Target.CAUSAL.value: ["ResearcherAgent"],
    Target.SESSION.value: ["Orchestrator"],
}


# ---------------------------------------------------------------------------
# Proposal data class
# ---------------------------------------------------------------------------


@dataclass
class Proposal:
    proposal_id: str = ""
    target: str = ""
    action: str = "update"
    data: Any = None
    agent_id: str = ""
    timestamp: float = field(default_factory=time.time)
    status: str = "pending"
    result: Any = None
    error: Optional[str] = None

    def __post_init__(self) -> None:
        if not self.proposal_id:
            self.proposal_id = f"prp_{uuid.uuid4().hex[:12]}"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "proposal_id": self.proposal_id,
            "target": self.target,
            "action": self.action,
            "data": self.data,
            "agent_id": self.agent_id,
            "timestamp": self.timestamp,
            "status": self.status,
            "result": self.result,
            "error": self.error,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Proposal":
        return cls(
            proposal_id=data.get("proposal_id", ""),
            target=data.get("target", ""),
            action=data.get("action", "update"),
            data=data.get("data"),
            agent_id=data.get("agent_id", ""),
            timestamp=float(data.get("timestamp", time.time())),
            status=data.get("status", "pending"),
            result=data.get("result"),
            error=data.get("error"),
        )


# ---------------------------------------------------------------------------
# Knowledge Manager
# ---------------------------------------------------------------------------


class KnowledgeManager:
    """Unified read/write API for all knowledge stores.

    Each knowledge *target* has designated *owners* that can write directly.
    Non-owner agents must submit proposals via ``propose()``.

    Two backends are supported:

    * **Redis** (production) — proposals are stored in a Redis list per target
      and proposal status is stored under ``proposal:<id>`` hashes.
    * **In-memory** (dev/test) — uses ``queue.Queue`` per target and a plain
      dict.  No external dependencies.

    Parameters:
        redis_url:
            Redis connection URL (e.g. ``redis://localhost:6379/0``).
            If ``None``, the environment variable ``REDIS_URL`` is consulted.
            If that is also unset and ``use_redis`` is ``True``, an attempt
            is made to connect to ``redis://localhost:6379/0``.
        use_redis:
            When ``True`` (default), the Redis backend is used (falling back
            gracefully to in-memory if no Redis server is reachable).
            When ``False``, the in-memory backend is used unconditionally.
    """

    def __init__(
        self,
        redis_url: Optional[str] = None,
        use_redis: bool = True,
    ) -> None:
        self._owners: Dict[str, List[str]] = dict(DEFAULT_OWNERS)
        self._stores: Dict[str, Any] = {}
        self._redis_url = redis_url or "redis://localhost:6379/0"
        self._use_redis = use_redis
        self._redis_client: Optional[Any] = None
        self._memory_queues: Dict[str, MemoryQueue] = {}
        self._memory_proposals: Dict[str, Dict[str, Any]] = {}

        if use_redis:
            self._init_redis()

        if not use_redis or self._redis_client is None:
            self._init_memory_queues()

    # ------------------------------------------------------------------
    # Store registration
    # ------------------------------------------------------------------

    def register_store(self, target: str, store_instance: Any) -> None:
        """Register a store module for *target*.

        ``read()`` and ``write()`` delegate to this store instance.
        """
        if target not in {t.value for t in Target}:
            raise ValueError(
                f"Unknown target '{target}'. "
                f"Valid: {[t.value for t in Target]}"
            )
        self._stores[target] = store_instance
        logger.debug("Registered store for target '%s': %s", target, type(store_instance).__name__)

    def get_store(self, target: str) -> Any:
        """Return the registered store instance for *target*."""
        store = self._stores.get(target)
        if store is None:
            raise ValueError(
                f"No store registered for target '{target}'. "
                "Call register_store() first."
            )
        return store

    # ------------------------------------------------------------------
    # Read / Write
    # ------------------------------------------------------------------

    def read(self, target: str, query: Any) -> Any:
        """Read directly from *target* store (no permission check)."""
        store = self.get_store(target)
        if hasattr(store, "get") and not isinstance(query, dict):
            return store.get(query)
        if hasattr(store, "find") and callable(getattr(store, "find")):
            return store.find(**query) if isinstance(query, dict) else store.find(query)
        return store.get(query)

    def write(self, target: str, data: Any, agent_id: str) -> Any:
        """Write directly to *target* if *agent_id* is an owner.

        Raises:
            PermissionError: If the agent is not an owner of this target.
        """
        owners = self._owners.get(target, [])
        if agent_id not in owners:
            raise PermissionError(
                f"Agent '{agent_id}' is not an owner of '{target}'. "
                f"Owners: {owners}. Use propose() instead."
            )
        store = self.get_store(target)
        if hasattr(store, "save"):
            return store.save(data)
        if hasattr(store, "add"):
            return store.add(data)
        raise TypeError(f"Store '{target}' has no save/add method")

    # ------------------------------------------------------------------
    # Proposal flow
    # ------------------------------------------------------------------

    def propose(
        self,
        target: str,
        data: Any,
        agent_id: str,
        action: str = "update",
    ) -> str:
        """Submit a proposal to *target* queue.

        The owner agent should poll and resolve the proposal.

        Returns:
            The proposal ID.
        """
        if target not in {t.value for t in Target}:
            raise ValueError(f"Unknown target '{target}'")

        proposal = Proposal(
            target=target,
            action=action,
            data=data,
            agent_id=agent_id,
        )

        if self._redis_client is not None:
            self._push_redis(proposal)
        else:
            self._push_memory(proposal)

        logger.info(
            "Proposal %s submitted by %s for target '%s' (action=%s)",
            proposal.proposal_id, agent_id, target, action,
        )
        return proposal.proposal_id

    def poll_proposals(
        self,
        agent_id: str,
        target: str,
        timeout: float = 0.0,
    ) -> List[Dict[str, Any]]:
        """Poll proposals from *target* queue.

        Only the owner of *target* may poll.  Returns a list of proposal
        dicts, or an empty list if the queue is empty.

        Args:
            agent_id: The polling agent's identifier.
            target: Target queue to poll.
            timeout: Maximum wait time in seconds (0 = no wait).

        Returns:
            List of proposal dicts.
        """
        owners = self._owners.get(target, [])
        if agent_id not in owners:
            logger.warning(
                "Agent '%s' is not an owner of '%s'; poll may be empty",
                agent_id, target,
            )
            return []

        proposals: List[Dict[str, Any]] = []

        if self._redis_client is not None:
            proposals = self._poll_redis(target, timeout)
        else:
            proposals = self._poll_memory(target, timeout)

        return proposals

    def resolve_proposal(
        self,
        proposal_id: str,
        accepted: bool,
        result: Any = None,
        error: Optional[str] = None,
    ) -> bool:
        """Mark a proposal as accepted or rejected.

        Returns True if the proposal was found and updated.
        """
        status = "accepted" if accepted else "rejected"

        if self._redis_client is not None:
            return self._resolve_redis(proposal_id, status, result, error)
        else:
            return self._resolve_memory(proposal_id, status, result, error)

    def get_proposal_status(self, proposal_id: str) -> Optional[Dict[str, Any]]:
        """Return the current status dict of a proposal, or None."""
        if self._redis_client is not None:
            return self._get_status_redis(proposal_id)
        else:
            return self._get_status_memory(proposal_id)

    # ------------------------------------------------------------------
    # Owners management
    # ------------------------------------------------------------------

    def set_owners(self, target: str, owners: List[str]) -> None:
        """Override owner list for a target."""
        self._owners[target] = list(owners)

    def get_owners(self, target: str) -> List[str]:
        """Return the list of owner agent IDs for *target*."""
        return list(self._owners.get(target, []))

    def is_owner(self, target: str, agent_id: str) -> bool:
        """Check if *agent_id* is an owner of *target*."""
        return agent_id in self._owners.get(target, [])

    # ------------------------------------------------------------------
    # Connection management
    # ------------------------------------------------------------------

    def close(self) -> None:
        if self._redis_client is not None:
            try:
                self._redis_client.close()
            except Exception:
                pass
            self._redis_client = None

    def __enter__(self) -> "KnowledgeManager":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Redis backend
    # ------------------------------------------------------------------

    def _init_redis(self) -> None:
        try:
            import redis as redis_module  # type: ignore[import-untyped]

            self._redis_client = redis_module.from_url(
                self._redis_url,
                socket_connect_timeout=3,
                socket_timeout=3,
                decode_responses=True,
            )
            self._redis_client.ping()
            logger.info("Connected to Redis at %s", self._redis_url)
        except Exception as exc:
            logger.warning(
                "Redis unavailable (%s); falling back to in-memory queues",
                exc,
            )
            self._redis_client = None

    def _queue_key(self, target: str) -> str:
        return f"proposal_queue:{target}"

    def _status_key(self, proposal_id: str) -> str:
        return f"proposal:{proposal_id}"

    def _push_redis(self, proposal: Proposal) -> None:
        assert self._redis_client is not None
        pipe = self._redis_client.pipeline()
        pipe.hset(
            self._status_key(proposal.proposal_id),
            mapping=proposal.to_dict(),
        )
        pipe.rpush(self._queue_key(proposal.target), proposal.proposal_id)
        pipe.execute()

    def _poll_redis(
        self, target: str, timeout: float
    ) -> List[Dict[str, Any]]:
        assert self._redis_client is not None
        proposals: List[Dict[str, Any]] = []
        max_items = 100
        for _ in range(max_items):
            if timeout > 0:
                result = self._redis_client.blpop(
                    self._queue_key(target), timeout=int(timeout)
                )
                if result is None:
                    break
                pid = result[1]
            else:
                pid = self._redis_client.lpop(self._queue_key(target))
                if pid is None:
                    break

            data = self._redis_client.hgetall(self._status_key(pid))
            if data:
                data.setdefault("status", "pending")
                proposals.append(data)
        return proposals

    def _resolve_redis(
        self,
        proposal_id: str,
        status: str,
        result: Any,
        error: Optional[str],
    ) -> bool:
        assert self._redis_client is not None
        key = self._status_key(proposal_id)
        if not self._redis_client.exists(key):
            return False
        self._redis_client.hset(key, "status", status)
        if result is not None:
            self._redis_client.hset(
                key, "result", json.dumps(result, ensure_ascii=False)
            )
        if error is not None:
            self._redis_client.hset(key, "error", error)
        return True

    def _get_status_redis(
        self, proposal_id: str
    ) -> Optional[Dict[str, Any]]:
        assert self._redis_client is not None
        data = self._redis_client.hgetall(self._status_key(proposal_id))
        if not data:
            return None
        return dict(data)

    # ------------------------------------------------------------------
    # In-memory backend
    # ------------------------------------------------------------------

    def _init_memory_queues(self) -> None:
        for target in Target:
            self._memory_queues[target.value] = MemoryQueue()
            self._memory_proposals[target.value] = {}

    def _push_memory(self, proposal: Proposal) -> None:
        queue = self._memory_queues.get(proposal.target)
        if queue is None:
            raise ValueError(f"Unknown target '{proposal.target}'")
        self._memory_proposals[proposal.target][proposal.proposal_id] = (
            proposal.to_dict()
        )
        queue.put(proposal.proposal_id)

    def _poll_memory(
        self, target: str, timeout: float
    ) -> List[Dict[str, Any]]:
        queue = self._memory_queues.get(target)
        if queue is None:
            return []

        proposals: List[Dict[str, Any]] = []
        try:
            pid = queue.get(block=timeout > 0, timeout=timeout if timeout > 0 else None)
        except Exception:
            return proposals

        data = self._memory_proposals.get(target, {}).get(pid)
        if data:
            proposals.append(data)

        while not queue.empty():
            try:
                pid = queue.get(block=False)
            except Exception:
                break
            data = self._memory_proposals.get(target, {}).get(pid)
            if data:
                proposals.append(data)

        return proposals

    def _resolve_memory(
        self,
        proposal_id: str,
        status: str,
        result: Any,
        error: Optional[str],
    ) -> bool:
        for target_data in self._memory_proposals.values():
            if proposal_id in target_data:
                target_data[proposal_id]["status"] = status
                if result is not None:
                    target_data[proposal_id]["result"] = result
                if error is not None:
                    target_data[proposal_id]["error"] = error
                return True
        return False

    def _get_status_memory(
        self, proposal_id: str
    ) -> Optional[Dict[str, Any]]:
        for target_data in self._memory_proposals.values():
            if proposal_id in target_data:
                return dict(target_data[proposal_id])
        return None
