"""Strategy Memory (L3) — stores and retrieves successful intervention strategies.

L3 in the hierarchical memory architecture (§7 of harmony_v5v.md).

Each *StrategyRecord* captures an intervention strategy that led to a
meaningful discriminative outcome between two hypotheses. These records can
be reused across campaigns and models to accelerate convergence.

Backends:
    - **Neo4j** (production): persistent graph storage.
    - **In-memory** (fallback / dev): plain Python dict.

Usage::

    from knowledge.strategy_memory import StrategyMemory, StrategyRecord

    sm = StrategyMemory(uri="bolt://localhost:7687",
                        user="neo4j", password="password")
    record = StrategyRecord(
        hypothesis_a_desc="model refuses if contains 'bomb'",
        hypothesis_b_desc="model accepts all prompts",
        transform_chain=["add_role_play", "to_lowercase"],
        outcome=1,                     # REFUSE observed
        discriminative_power=0.82,
        model_family="RLHF",
        campaign_id="camp_001",
    )
    rid = sm.save(record)

    # Warm-start next campaign
    hits = sm.find_strategies(model_family="RLHF", min_power=0.7)
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data class
# ---------------------------------------------------------------------------


@dataclass
class StrategyRecord:
    """A successful intervention strategy record.

    Attributes
    ----------
    id : str
        Unique identifier (auto-generated if empty).
    hypothesis_a_desc : str
        Description of hypothesis A involved in the discrimination.
    hypothesis_b_desc : str
        Description of hypothesis B involved in the discrimination.
    transform_chain : list of str
        Ordered list of transform names applied (e.g. ['add_role_play', 'to_lowercase']).
    outcome : int
        Observed outcome (0 = ACCEPT, 1 = REFUSE).
    discriminative_power : float
        Δ value [0, 1] quantifying how well this strategy discriminates A vs B.
    model_family : str
        Model family (e.g. "RLHF", "Constitutional AI", "Llama").
    campaign_id : str
        Campaign this record originates from.
    created_at : float
        Unix timestamp.
    metadata : dict
        Free-form metadata (e.g. prompt snippet, victim name).
    """

    hypothesis_a_desc: str = ""
    hypothesis_b_desc: str = ""
    transform_chain: List[str] = field(default_factory=list)
    outcome: int = 0
    discriminative_power: float = 0.0
    model_family: str = ""
    campaign_id: str = ""
    id: str = ""
    created_at: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.id:
            self.id = f"str_{uuid.uuid4().hex[:12]}"
        self.discriminative_power = max(0.0, min(1.0, float(self.discriminative_power)))
        self.outcome = int(self.outcome)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "hypothesis_a_desc": self.hypothesis_a_desc,
            "hypothesis_b_desc": self.hypothesis_b_desc,
            "transform_chain": self.transform_chain,
            "outcome": self.outcome,
            "discriminative_power": self.discriminative_power,
            "model_family": self.model_family,
            "campaign_id": self.campaign_id,
            "created_at": self.created_at,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StrategyRecord":
        return cls(
            id=data.get("id", ""),
            hypothesis_a_desc=data.get("hypothesis_a_desc", ""),
            hypothesis_b_desc=data.get("hypothesis_b_desc", ""),
            transform_chain=list(data.get("transform_chain", [])),
            outcome=int(data.get("outcome", 0)),
            discriminative_power=float(data.get("discriminative_power", 0.0)),
            model_family=data.get("model_family", ""),
            campaign_id=data.get("campaign_id", ""),
            created_at=float(data.get("created_at", time.time())),
            metadata=dict(data.get("metadata", {})),
        )


# ---------------------------------------------------------------------------
# Backend: Neo4j
# ---------------------------------------------------------------------------


class _Neo4jBackend:
    """Neo4j-backed storage for StrategyRecord nodes."""

    def __init__(self, uri: str, user: str, password: str, database: str) -> None:
        from neo4j import GraphDatabase, basic_auth  # type: ignore[import-untyped]
        from neo4j.exceptions import ServiceUnavailable  # type: ignore[import-untyped]

        self._database = database
        try:
            self._driver = GraphDatabase.driver(uri, auth=basic_auth(user, password))
            self._ensure_constraints()
        except ServiceUnavailable as exc:
            raise ConnectionError(
                f"Cannot connect to Neo4j at {uri}: {exc}"
            ) from exc

    def _session(self) -> Any:
        return self._driver.session(database=self._database)

    def _ensure_constraints(self) -> None:
        with self._session() as session:
            session.run(
                "CREATE CONSTRAINT IF NOT EXISTS "
                "FOR (s:StrategyRecord) REQUIRE s.id IS UNIQUE"
            )
            session.run(
                "CREATE INDEX IF NOT EXISTS FOR (s:StrategyRecord) "
                "ON (s.model_family)"
            )
            session.run(
                "CREATE INDEX IF NOT EXISTS FOR (s:StrategyRecord) "
                "ON (s.discriminative_power)"
            )
            session.run(
                "CREATE INDEX IF NOT EXISTS FOR (s:StrategyRecord) "
                "ON (s.campaign_id)"
            )

    def save(self, record: StrategyRecord) -> str:
        with self._session() as session:
            session.run(
                """
                MERGE (s:StrategyRecord {id: $id})
                SET s.hypothesis_a_desc    = $hypothesis_a_desc,
                    s.hypothesis_b_desc    = $hypothesis_b_desc,
                    s.transform_chain      = $transform_chain,
                    s.outcome              = $outcome,
                    s.discriminative_power = $discriminative_power,
                    s.model_family         = $model_family,
                    s.campaign_id          = $campaign_id,
                    s.created_at           = $created_at,
                    s.metadata             = $metadata
                """,
                id=record.id,
                hypothesis_a_desc=record.hypothesis_a_desc,
                hypothesis_b_desc=record.hypothesis_b_desc,
                transform_chain=json.dumps(record.transform_chain, ensure_ascii=False),
                outcome=record.outcome,
                discriminative_power=record.discriminative_power,
                model_family=record.model_family,
                campaign_id=record.campaign_id,
                created_at=record.created_at,
                metadata=json.dumps(record.metadata, ensure_ascii=False),
            )
        logger.debug("Saved StrategyRecord %s (power=%.2f)", record.id, record.discriminative_power)
        return record.id

    def find(
        self,
        model_family: Optional[str] = None,
        min_power: float = 0.5,
        campaign_id: Optional[str] = None,
        limit: int = 50,
    ) -> List[StrategyRecord]:
        where_parts = ["s.discriminative_power >= $min_power"]
        params: Dict[str, Any] = {"min_power": min_power, "limit": limit}
        if model_family:
            where_parts.append("s.model_family = $model_family")
            params["model_family"] = model_family
        if campaign_id:
            where_parts.append("s.campaign_id = $campaign_id")
            params["campaign_id"] = campaign_id
        where_clause = "WHERE " + " AND ".join(where_parts)
        with self._session() as session:
            result = session.run(
                f"MATCH (s:StrategyRecord) {where_clause} "
                f"RETURN s ORDER BY s.discriminative_power DESC LIMIT $limit",
                **params,
            )
            return [self._node_to_record(rec["s"]) for rec in result]

    def get(self, record_id: str) -> Optional[StrategyRecord]:
        with self._session() as session:
            result = session.run(
                "MATCH (s:StrategyRecord {id: $id}) RETURN s",
                id=record_id,
            )
            record = result.single()
            if record is None:
                return None
            return self._node_to_record(record["s"])

    def delete(self, record_id: str) -> bool:
        with self._session() as session:
            result = session.run(
                "MATCH (s:StrategyRecord {id: $id}) DELETE s "
                "RETURN count(s) AS deleted",
                id=record_id,
            )
            rec = result.single()
            return rec is not None and rec["deleted"] > 0

    def close(self) -> None:
        if self._driver is not None:
            self._driver.close()
            self._driver = None  # type: ignore[assignment]

    @staticmethod
    def _node_to_record(node: Any) -> StrategyRecord:
        chain_raw = node.get("transform_chain", "[]")
        try:
            chain = json.loads(chain_raw)
        except (json.JSONDecodeError, TypeError):
            chain = []
        meta_raw = node.get("metadata", "{}")
        try:
            metadata = json.loads(meta_raw)
        except (json.JSONDecodeError, TypeError):
            metadata = {}
        return StrategyRecord(
            id=node.get("id", ""),
            hypothesis_a_desc=node.get("hypothesis_a_desc", ""),
            hypothesis_b_desc=node.get("hypothesis_b_desc", ""),
            transform_chain=chain,
            outcome=int(node.get("outcome", 0)),
            discriminative_power=float(node.get("discriminative_power", 0.0)),
            model_family=node.get("model_family", ""),
            campaign_id=node.get("campaign_id", ""),
            created_at=float(node.get("created_at", time.time())),
            metadata=metadata,
        )


# ---------------------------------------------------------------------------
# Backend: In-memory fallback
# ---------------------------------------------------------------------------


class _InMemoryBackend:
    """Pure Python dict backend for dev / testing without Neo4j."""

    def __init__(self) -> None:
        self._store: Dict[str, StrategyRecord] = {}

    def save(self, record: StrategyRecord) -> str:
        self._store[record.id] = record
        logger.debug("InMemory: saved StrategyRecord %s", record.id)
        return record.id

    def find(
        self,
        model_family: Optional[str] = None,
        min_power: float = 0.5,
        campaign_id: Optional[str] = None,
        limit: int = 50,
    ) -> List[StrategyRecord]:
        results = []
        for rec in self._store.values():
            if rec.discriminative_power < min_power:
                continue
            if model_family and rec.model_family != model_family:
                continue
            if campaign_id and rec.campaign_id != campaign_id:
                continue
            results.append(rec)
        results.sort(key=lambda r: r.discriminative_power, reverse=True)
        return results[:limit]

    def get(self, record_id: str) -> Optional[StrategyRecord]:
        return self._store.get(record_id)

    def delete(self, record_id: str) -> bool:
        if record_id in self._store:
            del self._store[record_id]
            return True
        return False

    def close(self) -> None:
        pass


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


class StrategyMemory:
    """L3 Strategy Memory — persistent store for successful intervention strategies.

    Parameters
    ----------
    uri : str
        Neo4j bolt URI. If ``None`` or Neo4j is unavailable, falls back to
        in-memory storage.
    user : str
    password : str
    database : str
    use_neo4j : bool
        When ``False``, always uses in-memory backend.
    """

    def __init__(
        self,
        uri: Optional[str] = "bolt://localhost:7687",
        user: str = "neo4j",
        password: str = "password",
        database: str = "neo4j",
        use_neo4j: bool = True,
    ) -> None:
        self._backend: Any = None
        if use_neo4j and uri:
            try:
                self._backend = _Neo4jBackend(uri, user, password, database)
                logger.info("StrategyMemory: connected to Neo4j at %s", uri)
            except Exception as exc:
                logger.warning(
                    "StrategyMemory: Neo4j unavailable (%s); using in-memory fallback",
                    exc,
                )
        if self._backend is None:
            self._backend = _InMemoryBackend()
            logger.info("StrategyMemory: using in-memory backend")

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    def save(self, record: StrategyRecord) -> str:
        """Persist a StrategyRecord.  Returns its ID."""
        return self._backend.save(record)

    def get(self, record_id: str) -> Optional[StrategyRecord]:
        """Retrieve a record by ID, or None."""
        return self._backend.get(record_id)

    def find_strategies(
        self,
        model_family: Optional[str] = None,
        min_power: float = 0.5,
        campaign_id: Optional[str] = None,
        limit: int = 50,
    ) -> List[StrategyRecord]:
        """Find high-discriminative-power strategies.

        Parameters
        ----------
        model_family : str, optional
            Filter by model family (e.g. "RLHF").
        min_power : float
            Minimum discriminative power threshold (default 0.5).
        campaign_id : str, optional
            Restrict to a specific prior campaign.
        limit : int
            Maximum number of results (default 50).

        Returns
        -------
        List of StrategyRecord sorted by discriminative_power descending.
        """
        return self._backend.find(
            model_family=model_family,
            min_power=min_power,
            campaign_id=campaign_id,
            limit=limit,
        )

    def suggest_transforms(
        self,
        model_family: Optional[str] = None,
        min_power: float = 0.5,
        top_k: int = 5,
    ) -> List[List[str]]:
        """Return the top-k transform chains from prior strategies.

        Useful for warm-starting the StrategistAgent's candidate set.

        Returns
        -------
        List of transform_chain lists, ordered by discriminative_power.
        """
        records = self.find_strategies(
            model_family=model_family,
            min_power=min_power,
            limit=top_k * 3,  # over-fetch to deduplicate
        )
        seen: set = set()
        chains: List[List[str]] = []
        for rec in records:
            key = tuple(rec.transform_chain)
            if key not in seen:
                seen.add(key)
                chains.append(rec.transform_chain)
            if len(chains) >= top_k:
                break
        return chains

    def delete(self, record_id: str) -> bool:
        """Delete a record by ID.  Returns True if found."""
        return self._backend.delete(record_id)

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Release backend resources."""
        if self._backend is not None:
            self._backend.close()

    def __enter__(self) -> "StrategyMemory":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()
