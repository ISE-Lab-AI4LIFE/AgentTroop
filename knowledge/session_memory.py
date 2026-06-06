"""Session Memory (L2) — Redis-backed campaign state cache.

Each campaign is stored as a Redis hash with key ``session:{campaign_id}``
and an automatic TTL.  Provides atomic increment operations and hypothesis
list management via Redis lists.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_SESSION_PREFIX = "session:"
_HYP_LIST_KEY = "hypothesis_ids"
_META_KEY = "metadata"
_DEFAULT_TTL = 86400  # 24 hours


def _session_key(campaign_id: str) -> str:
    return f"{_SESSION_PREFIX}{campaign_id}"


def _hyp_list_key(campaign_id: str) -> str:
    return f"{_SESSION_PREFIX}{campaign_id}:hypotheses"


class SessionMemory:
    """Redis-backed cache for campaign session state (L2).

    Each campaign is stored as a hash under ``session:{campaign_id}`` with
    automatic TTL expiry.  Hypothesis IDs are stored in a separate Redis
    list ``session:{campaign_id}:hypotheses`` for atomic push/remove.

    Parameters:
        redis_url: Redis connection URL (default ``redis://localhost:6379/0``).
        ttl: Time-to-live in seconds (default 86400 = 24 hours).
    """

    def __init__(
        self,
        redis_url: str = "redis://localhost:6379/0",
        ttl: int = _DEFAULT_TTL,
    ) -> None:
        import redis as redis_module  # type: ignore[import-untyped]

        self.client = redis_module.from_url(
            redis_url,
            decode_responses=True,
            socket_connect_timeout=5,
            socket_timeout=5,
        )
        self.client.ping()
        self.ttl = max(60, int(ttl))
        logger.info("SessionMemory connected to %s (TTL=%ds)", redis_url, self.ttl)

    def close(self) -> None:
        try:
            self.client.close()
        except Exception:
            pass

    def __enter__(self) -> "SessionMemory":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Core CRUD
    # ------------------------------------------------------------------

    def create_session(
        self,
        campaign_id: str,
        target_model: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Create a new campaign session.  Returns False if already exists."""
        key = _session_key(campaign_id)
        existed = self.client.exists(key)
        if existed:
            return False

        now = time.time()
        mapping: Dict[str, Any] = {
            "campaign_id": campaign_id,
            "target_model": target_model,
            "current_best_program_id": "",
            "current_best_accuracy": "0.0",
            "iteration": "0",
            "intervention_count": "0",
            "status": "running",
            "started_at": str(now),
            "updated_at": str(now),
            _META_KEY: json.dumps(metadata or {}, ensure_ascii=False),
        }
        pipe = self.client.pipeline()
        pipe.hset(key, mapping=mapping)
        pipe.expire(key, self.ttl)
        pipe.execute()
        logger.info("Session created: campaign=%s model=%s", campaign_id, target_model)
        return True

    def get_session(self, campaign_id: str) -> Optional[Dict[str, Any]]:
        """Return the full session dict, or None if not found."""
        key = _session_key(campaign_id)
        data = self.client.hgetall(key)
        if not data:
            return None

        data[_META_KEY] = json.loads(data.get(_META_KEY, "{}"))
        data["current_best_accuracy"] = float(data.get("current_best_accuracy", 0.0))
        data["iteration"] = int(data.get("iteration", 0))
        data["intervention_count"] = int(data.get("intervention_count", 0))
        data["started_at"] = float(data.get("started_at", 0.0))
        data["updated_at"] = float(data.get("updated_at", 0.0))

        hyp_ids = self.client.lrange(_hyp_list_key(campaign_id), 0, -1)
        data[_HYP_LIST_KEY] = list(hyp_ids)
        return data

    def update_session(
        self, campaign_id: str, updates: Dict[str, Any]
    ) -> bool:
        """Update one or more fields of a session.  Returns False if not found."""
        key = _session_key(campaign_id)
        if not self.client.exists(key):
            return False

        mapping: Dict[str, Any] = {}
        for k, v in updates.items():
            if k == _META_KEY and isinstance(v, dict):
                mapping[k] = json.dumps(v, ensure_ascii=False)
            elif k in ("current_best_accuracy",):
                mapping[k] = str(v)
            elif k in ("iteration", "intervention_count"):
                mapping[k] = str(v)
            elif k in (
                "campaign_id", "target_model", "status",
                "current_best_program_id",
            ):
                mapping[k] = v
        mapping["updated_at"] = str(time.time())

        pipe = self.client.pipeline()
        pipe.hset(key, mapping=mapping)
        pipe.expire(key, self.ttl)
        pipe.execute()
        return True

    def delete_session(self, campaign_id: str) -> bool:
        """Delete a session and its hypothesis list.  Returns False if not found."""
        key = _session_key(campaign_id)
        hyp_key = _hyp_list_key(campaign_id)
        if not self.client.exists(key):
            return False
        pipe = self.client.pipeline()
        pipe.delete(key)
        pipe.delete(hyp_key)
        pipe.execute()
        logger.info("Session deleted: campaign=%s", campaign_id)
        return True

    # ------------------------------------------------------------------
    # Atomic increments
    # ------------------------------------------------------------------

    def increment_iteration(self, campaign_id: str) -> int:
        """Atomically increment the iteration counter.  Returns new value."""
        key = _session_key(campaign_id)
        new_val = self.client.hincrby(key, "iteration", 1)
        self.client.hset(key, "updated_at", str(time.time()))
        self.client.expire(key, self.ttl)
        return new_val

    def increment_intervention_count(
        self, campaign_id: str, delta: int = 1
    ) -> int:
        """Atomically increment the intervention counter.  Returns new value."""
        key = _session_key(campaign_id)
        new_val = self.client.hincrby(key, "intervention_count", delta)
        self.client.hset(key, "updated_at", str(time.time()))
        self.client.expire(key, self.ttl)
        return new_val

    # ------------------------------------------------------------------
    # Best program
    # ------------------------------------------------------------------

    def set_best_program(
        self, campaign_id: str, program_id: str, accuracy: float,
    ) -> bool:
        """Set the best program ID and accuracy.  Returns False if session not found."""
        key = _session_key(campaign_id)
        if not self.client.exists(key):
            return False
        pipe = self.client.pipeline()
        pipe.hset(key, "current_best_program_id", program_id)
        pipe.hset(key, "current_best_accuracy", str(accuracy))
        pipe.hset(key, "updated_at", str(time.time()))
        pipe.expire(key, self.ttl)
        pipe.execute()
        return True

    # ------------------------------------------------------------------
    # Hypothesis list management
    # ------------------------------------------------------------------

    def add_hypothesis(self, campaign_id: str, hypothesis_id: str) -> bool:
        """Add a hypothesis ID to the campaign's list.  Returns True."""
        hyp_key = _hyp_list_key(campaign_id)
        pipe = self.client.pipeline()
        pipe.rpush(hyp_key, hypothesis_id)
        pipe.expire(hyp_key, self.ttl)
        pipe.execute()
        key = _session_key(campaign_id)
        self.client.hset(key, "updated_at", str(time.time()))
        self.client.expire(key, self.ttl)
        return True

    def remove_hypothesis(self, campaign_id: str, hypothesis_id: str) -> bool:
        """Remove a hypothesis ID from the campaign's list.  Returns True."""
        hyp_key = _hyp_list_key(campaign_id)
        self.client.lrem(hyp_key, 0, hypothesis_id)
        key = _session_key(campaign_id)
        self.client.hset(key, "updated_at", str(time.time()))
        return True

    def list_hypotheses(self, campaign_id: str) -> List[str]:
        """Return all hypothesis IDs for a campaign."""
        return list(self.client.lrange(_hyp_list_key(campaign_id), 0, -1))

    # ------------------------------------------------------------------
    # Status management
    # ------------------------------------------------------------------

    def set_status(self, campaign_id: str, status: str) -> bool:
        """Set the session status.  Returns False if session not found."""
        key = _session_key(campaign_id)
        if not self.client.exists(key):
            return False
        pipe = self.client.pipeline()
        pipe.hset(key, "status", status)
        pipe.hset(key, "updated_at", str(time.time()))
        pipe.expire(key, self.ttl)
        pipe.execute()
        return True

    # ------------------------------------------------------------------
    # Session listing
    # ------------------------------------------------------------------

    def list_active_sessions(self) -> List[str]:
        """Return campaign IDs for all active (non-expired) sessions."""
        cursor: Any = 0
        campaign_ids: List[str] = []
        pattern = f"{_SESSION_PREFIX}*"
        while True:
            cursor, keys = self.client.scan(
                cursor=cursor, match=pattern, count=100,
            )
            for key in keys:
                cid = key[len(_SESSION_PREFIX):]
                if ":" not in cid:
                    campaign_ids.append(cid)
            if cursor == 0:
                break
        return campaign_ids

    # ------------------------------------------------------------------
    # Session existence
    # ------------------------------------------------------------------

    def session_exists(self, campaign_id: str) -> bool:
        """Check if a session exists."""
        return bool(self.client.exists(_session_key(campaign_id)))
