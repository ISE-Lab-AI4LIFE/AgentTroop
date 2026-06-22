"""Scientific Memory — transferable safety theories for HARMONY-X.

L6 in the hierarchical memory architecture. Stores abstract theories
extracted from confirmed defense programs. Each theory encodes a
reusable safety pattern that can be transferred across model families
(RLHF, Constitutional AI, etc.) to bootstrap new reverse-engineering
campaigns.

Features:
- Conditions stored as dynamic Neo4j properties for fast Cypher-level filtering.
- Full version history (each save creates a new version node linked via
  ``NEXT_VERSION``).  ``get_theory()`` always returns the latest version.
- Pattern substring search via ``find_theories_by_pattern()``.
- **Auto-compact:** ``compact_if_needed()``, ``compact_older_than()``,
  and optionally a background thread via ``set_auto_compact_enabled()``.
- Export/import with optional version history.
- Context-manager support.
- Logging and monitoring (``get_version_stats()``).

Requires a running Neo4j instance (community edition is sufficient).
Install the driver with ``pip install neo4j``.

Quick-start — local Neo4j (Docker):
    docker run --rm -p 7687:7687 -e NEO4J_AUTH=neo4j/password neo4j:5

Usage::

    from knowledge.scientific_memory import ScientificMemory, Theory

    memory = ScientificMemory("bolt://localhost:7687", "neo4j", "password")
    theory = Theory(
        pattern="IF contains(decode_rot13(x), 'bomb') THEN REFUSE",
        conditions={"model_family": "RLHF", "cipher": "rot13"},
        confidence=0.92,
        provenance=["ep_123", "ep_456"],
    )
    tid = memory.save_theory(theory)
    retrieved = memory.get_theory(tid)
    assert retrieved.confidence == 0.92

    theories = memory.find_theories(
        conditions={"model_family": "RLHF"}, min_confidence=0.9
    )
    assert len(theories) >= 1

    memory.compact_if_needed(keep_versions=10, max_versions_before_compact=20)

    stats = memory.get_version_stats(tid)
    print(stats)

    memory.close()
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

from neo4j import GraphDatabase, Session, basic_auth  # type: ignore[import-untyped]
from neo4j.exceptions import ServiceUnavailable  # type: ignore[import-untyped]

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logger = logging.getLogger(__name__)

# Known condition keys that appear frequently — used for index hints.
# Users may add custom keys at any time via the conditions dict.
_KNOWN_CONDITION_KEYS: Set[str] = {
    "model_family",
    "cipher",
    "model_architecture",
    "training_paradigm",
    "safety_level",
}

# Neo4j-internal property keys on the Theory node — these are
# never part of user-supplied conditions.
_RESERVED_PROPS: Set[str] = {
    "id",
    "version",
    "pattern",
    "confidence",
    "provenance",
    "created_at",
    "updated_at",
    "conditions_json",
    "metadata",
}


# ---------------------------------------------------------------------------
# Data class
# ---------------------------------------------------------------------------


@dataclass
class Theory:
    """An abstract safety theory that can be transferred across models.

    Each save creates a new *version*; ``get_theory()`` always returns
    the latest one.

    Attributes:
        id: Unique identifier (UUID, auto-generated if empty).  Multiple
            versions share the same id.
        pattern: The theory as a logic rule or program fragment.
        conditions: Conditions under which the theory applies
            (e.g. ``{"model_family": "RLHF", "cipher": "rot13"}``).
        confidence: Confidence score in ``[0, 1]``.
        provenance: List of intervention or episode IDs that support
            this theory.
        version: Version number (1-based).  Auto-managed; do not set
            manually when saving.
        created_at: Unix timestamp of first creation.
        updated_at: Unix timestamp of this version.
        metadata: Free-form metadata.
    """

    pattern: str
    conditions: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.0
    provenance: List[str] = field(default_factory=list)
    id: str = ""
    version: int = 1
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.id:
            self.id = f"thr_{uuid.uuid4().hex[:12]}"
        self.confidence = max(0.0, min(1.0, float(self.confidence)))
        self.version = max(1, int(self.version))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "pattern": self.pattern,
            "conditions": self.conditions,
            "confidence": self.confidence,
            "provenance": self.provenance,
            "version": self.version,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Theory":
        return cls(
            id=data.get("id", ""),
            pattern=data.get("pattern", ""),
            conditions=dict(data.get("conditions", {})),
            confidence=float(data.get("confidence", 0.0)),
            provenance=list(data.get("provenance", [])),
            version=int(data.get("version", 1)),
            created_at=float(data.get("created_at", time.time())),
            updated_at=float(data.get("updated_at", time.time())),
            metadata=dict(data.get("metadata", {})),
        )


# ---------------------------------------------------------------------------
# Neo4j-backed store
# ---------------------------------------------------------------------------


class ScientificMemory:
    """Neo4j-backed store for versioned, transferable safety theories.

    Parameters:
        uri: Bolt URI (e.g. ``bolt://localhost:7687``).
        user: Neo4j username.
        password: Neo4j password.
        database: Database name (defaults to ``neo4j``).
    """

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

        # Auto-compact background thread state
        self._auto_compact_enabled: bool = False
        self._auto_compact_keep_versions: int = 10
        self._auto_compact_check_interval: float = 3600.0  # seconds
        self._auto_compact_timer: Optional[threading.Timer] = None
        self._auto_compact_lock: threading.Lock = threading.Lock()

    # ------------------------------------------------------------------
    # Connection management
    # ------------------------------------------------------------------

    def _get_driver(self) -> Any:
        if self._driver is None:
            try:
                if os.environ.get("HX_NEO4J_AUTH_DISABLED", "0") == "1":
                    self._driver = GraphDatabase.driver(self.uri)
                else:
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
        self.disable_auto_compact()
        if self._driver is not None:
            self._driver.close()
            self._driver = None

    def __enter__(self) -> "ScientificMemory":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Schema initialisation
    # ------------------------------------------------------------------

    def _ensure_constraints(self) -> None:
        """Create indexes and constraints for the Theory node.

        Safe to call repeatedly (IF NOT EXISTS semantics).
        Drops any legacy single-property ``id`` constraint if present.
        """
        with self._session() as session:
            # Migrate away from legacy single-property constraint
            result = session.run(
                "SHOW CONSTRAINTS WHERE type = 'NODE_PROPERTY_UNIQUENESS' "
                "AND labelsOrTypes = ['Theory'] "
                "AND properties = ['id']"
            )
            for record in result:
                name = record.get("name", "")
                if name:
                    session.run(f"DROP CONSTRAINT `{name}`")

            session.run(
                "CREATE CONSTRAINT IF NOT EXISTS "
                "FOR (t:Theory) REQUIRE (t.id, t.version) IS UNIQUE"
            )
            session.run(
                "CREATE INDEX IF NOT EXISTS FOR (t:Theory) "
                "ON (t.confidence)"
            )
            session.run(
                "CREATE INDEX IF NOT EXISTS FOR (t:Theory) "
                "ON (t.updated_at)"
            )
            session.run(
                "CREATE INDEX IF NOT EXISTS FOR (t:Theory) "
                "ON (t.created_at)"
            )
            # Indexes for well-known condition keys
            for key in _KNOWN_CONDITION_KEYS:
                safe_key = _safe_cypher_prop(key)
                session.run(
                    f"CREATE INDEX IF NOT EXISTS FOR (t:Theory) "
                    f"ON (t.`{safe_key}`)"
                )

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    def save_theory(self, theory: Theory) -> str:
        """Persist a theory as a new version. Returns its id.

        The first save creates version 1.  Subsequent saves for the same
        ``id`` create a new version and link it via ``NEXT_VERSION``.
        ``get_theory()`` will always return the latest version.
        """
        theory.confidence = max(0.0, min(1.0, float(theory.confidence)))
        theory.version = max(1, int(theory.version))
        now = time.time()
        theory.updated_at = now

        # Determine next version number
        existing = self._get_latest_node(theory.id)
        if existing is not None:
            theory.version = existing.get("version", 0) + 1
            theory.created_at = float(existing.get("created_at", now))
        else:
            theory.version = 1
            theory.created_at = now
            theory.updated_at = now

        conditions_map = {
            k: v for k, v in theory.conditions.items()
            if k not in _RESERVED_PROPS
        }

        with self._session() as session:
            tx = session.begin_transaction()
            try:
                # Create the new version node
                tx.run(
                    """CREATE (t:Theory {
                        id: $id,
                        version: $version,
                        pattern: $pattern,
                        confidence: $confidence,
                        provenance: $provenance,
                        created_at: $created_at,
                        updated_at: $updated_at,
                        conditions_json: $conditions_json,
                        metadata: $metadata
                    })
                    SET t += $conditions_map
                    """,
                    id=theory.id,
                    version=theory.version,
                    pattern=theory.pattern,
                    confidence=theory.confidence,
                    provenance=json.dumps(
                        theory.provenance, ensure_ascii=False
                    ),
                    created_at=theory.created_at,
                    updated_at=theory.updated_at,
                    conditions_json=json.dumps(
                        theory.conditions, ensure_ascii=False
                    ),
                    metadata=json.dumps(
                        theory.metadata, ensure_ascii=False
                    ),
                    conditions_map=conditions_map,
                )

                # Link from previous latest version
                if existing is not None:
                    prev_id = existing.element_id
                    tx.run(
                        f"""MATCH (prev)
                            WHERE elementId(prev) = $prev_id
                            MATCH (cur:Theory {{
                                id: $id, version: $version
                            }})
                            MERGE (prev)-[:NEXT_VERSION]->(cur)
                        """,
                        prev_id=prev_id,
                        id=theory.id,
                        version=theory.version,
                    )
                tx.commit()
            except Exception:
                tx.rollback()
                raise
        logger.debug("Saved theory %s version %d", theory.id, theory.version)
        return theory.id

    def get_theory(self, theory_id: str) -> Optional[Theory]:
        """Retrieve the **latest** version of a theory, or None."""
        node = self._get_latest_node(theory_id)
        if node is None:
            return None
        return self._node_to_theory(node)

    def get_theory_version(
        self, theory_id: str, version: int
    ) -> Optional[Theory]:
        """Retrieve a specific version of a theory, or None."""
        with self._session() as session:
            result = session.run(
                "MATCH (t:Theory {id: $id, version: $version}) RETURN t",
                id=theory_id,
                version=version,
            )
            record = result.single()
            if record is None:
                return None
            return self._node_to_theory(record["t"])

    def find_theories(
        self,
        conditions: Optional[Dict[str, Any]] = None,
        min_confidence: float = 0.0,
        target_model: Optional[str] = None,
    ) -> List[Theory]:
        """Find theories matching conditions and minimum confidence.

        Fix 6B: Added optional ``target_model`` parameter for backward
        compatibility with callers that pass ``target_model`` as a
        keyword argument.

        Filtering is performed at the Cypher level using dynamic
        properties, so it is efficient even with thousands of theories.
        Only the **latest** version of each theory is returned.
        """
        where_parts: List[str] = []
        params: Dict[str, Any] = {"min_conf": min_confidence}

        if conditions:
            for key, value in conditions.items():
                safe_key = _safe_cypher_prop(key)
                param_key = f"cond_{safe_key}"
                where_parts.append(f"t.`{safe_key}` = ${param_key}")
                params[param_key] = value

        if target_model is not None:
            where_parts.append("t.target_model = $target_model")
            params["target_model"] = target_model

        cond_clause = (
            f"AND {' AND '.join(where_parts)}" if where_parts else ""
        )

        with self._session() as session:
            result = session.run(
                f"""MATCH (t:Theory)
                    WHERE t.confidence >= $min_conf
                    AND NOT EXISTS ((t)-[:NEXT_VERSION]->())
                    {cond_clause}
                    RETURN t
                    ORDER BY t.created_at DESC
                """,
                **params,
            )
            return [
                self._node_to_theory(record["t"]) for record in result
            ]

    def find_theories_by_pattern(
        self,
        keyword: str,
        case_sensitive: bool = False,
        min_confidence: float = 0.0,
    ) -> List[Theory]:
        """Find latest theories whose pattern contains *keyword*.

        Uses Cypher ``CONTAINS``.  When ``case_sensitive=False`` the
        comparison is lowered on both sides.
        """
        if case_sensitive:
            template = (
                "MATCH (t:Theory) "
                "WHERE t.pattern CONTAINS $keyword "
                "AND t.confidence >= $min_conf "
                "AND NOT EXISTS ((t)-[:NEXT_VERSION]->()) "
                "RETURN t ORDER BY t.created_at DESC"
            )
        else:
            template = (
                "MATCH (t:Theory) "
                "WHERE toLower(t.pattern) CONTAINS toLower($keyword) "
                "AND t.confidence >= $min_conf "
                "AND NOT EXISTS ((t)-[:NEXT_VERSION]->()) "
                "RETURN t ORDER BY t.created_at DESC"
            )

        with self._session() as session:
            result = session.run(
                template,
                keyword=keyword,
                min_conf=min_confidence,
            )
            return [
                self._node_to_theory(record["t"]) for record in result
            ]

    def update_confidence(
        self, theory_id: str, new_confidence: float
    ) -> bool:
        """Create a new version with updated confidence.

        Returns True if the theory existed and a new version was created.
        """
        existing = self._get_latest_node(theory_id)
        if existing is None:
            return False
        theory = self._node_to_theory(existing)
        theory.confidence = new_confidence
        self.save_theory(theory)
        return True

    def delete_theory(self, theory_id: str) -> bool:
        """Delete **all** versions of a theory. Returns True if any
        version was deleted."""
        with self._session() as session:
            result = session.run(
                "MATCH (t:Theory {id: $id}) "
                "DETACH DELETE t "
                "RETURN count(t) AS deleted",
                id=theory_id,
            )
            record = result.single()
            return record is not None and record["deleted"] > 0

    def delete_all(self) -> int:
        """Delete all Theory nodes (all versions). Returns count."""
        with self._session() as session:
            result = session.run(
                "MATCH (t:Theory) DETACH DELETE t "
                "RETURN count(t) AS deleted"
            )
            record = result.single()
            return record["deleted"] if record else 0

    # ------------------------------------------------------------------
    # Auto-compact
    # ------------------------------------------------------------------

    def compact_theory(
        self, theory_id: str, keep_versions: int = 10
    ) -> int:
        """Remove old versions of a theory, keeping the *keep_versions*
        newest ones.

        At least one version is always retained (if ``keep_versions < 1``
        it is treated as 1).  Returns the number of versions deleted.
        """
        keep = max(1, int(keep_versions))
        t0 = time.time()
        with self._session() as session:
            result = session.run(
                """MATCH (t:Theory {id: $id})
                   WITH t ORDER BY t.version ASC
                   WITH collect(t) AS versions
                   WITH versions[0..size(versions)-$keep]
                       AS to_delete
                   UNWIND to_delete AS old
                   DETACH DELETE old
                   RETURN count(old) AS deleted
                """,
                id=theory_id,
                keep=keep,
            )
            record = result.single()
            deleted = record["deleted"] if record else 0
        elapsed = time.time() - t0
        if deleted > 0:
            logger.info(
                "Compact theory %s: deleted %d versions in %.2fs",
                theory_id, deleted, elapsed,
            )
        return deleted

    def compact_all(self, keep_versions: int = 10) -> Dict[str, int]:
        """Compact every theory that has more than *keep_versions*
        versions.

        Uses a single transaction to identify candidates, then compacts
        each individually with its own transaction.

        Returns a dict mapping ``theory_id → number_of_deleted_versions``.
        """
        keep = max(1, int(keep_versions))
        t0 = time.time()
        with self._session() as session:
            result = session.run(
                """MATCH (t:Theory)
                   WITH t.id AS tid, count(t) AS cnt
                   WHERE cnt > $keep
                   RETURN tid, cnt - $keep AS excess
                """,
                keep=keep,
            )
            candidates = [(r["tid"], r["excess"]) for r in result]

        summary: Dict[str, int] = {}
        for tid, _ in candidates:
            deleted = self.compact_theory(tid, keep_versions=keep)
            if deleted > 0:
                summary[tid] = deleted

        elapsed = time.time() - t0
        if summary:
            logger.info(
                "Compact all: %d theories compacted in %.2fs",
                len(summary), elapsed,
            )
        return summary

    def compact_if_needed(
        self,
        keep_versions: int = 10,
        max_versions_before_compact: int = 20,
    ) -> Dict[str, int]:
        """Auto-trigger compact for theories exceeding a version threshold.

        Queries all theories whose version count exceeds
        *max_versions_before_compact* and compacts them down to
        *keep_versions*.

        Returns a dict of ``{theory_id: deleted_count}``.
        """
        keep = max(1, int(keep_versions))
        threshold = max(keep + 1, int(max_versions_before_compact))
        t0 = time.time()

        with self._session() as session:
            result = session.run(
                """MATCH (t:Theory)
                   WITH t.id AS tid, count(t) AS cnt
                   WHERE cnt > $threshold
                   RETURN tid, cnt - $keep AS excess
                """,
                keep=keep,
                threshold=threshold,
            )
            candidates = [(r["tid"], r["excess"]) for r in result]

        summary: Dict[str, int] = {}
        for tid, _ in candidates:
            deleted = self.compact_theory(tid, keep_versions=keep)
            if deleted > 0:
                summary[tid] = deleted

        elapsed = time.time() - t0
        if summary:
            logger.info(
                "Compact if needed: %d theories compacted in %.2fs",
                len(summary), elapsed,
            )
        else:
            logger.debug("Compact if needed: no theories exceeded threshold")
        return summary

    def compact_older_than(
        self,
        days: int,
        keep_versions: int = 1,
    ) -> Dict[str, int]:
        """Delete old versions of theories, keeping at least
        *keep_versions* newest.

        Versions whose ``updated_at`` is older than *days* are deleted.
        At least *keep_versions* versions per theory are always retained.

        Returns a dict of ``{theory_id: deleted_count}``.
        """
        keep = max(1, int(keep_versions))
        cutoff = time.time() - days * 86400
        t0 = time.time()

        with self._session() as session:
            # Get theories with at least one version old enough
            result = session.run(
                """MATCH (t:Theory)
                   WHERE t.updated_at < $cutoff
                   RETURN DISTINCT t.id AS tid
                """,
                cutoff=cutoff,
            )
            candidate_ids = [r["tid"] for r in result]

        summary: Dict[str, int] = {}
        for tid in candidate_ids:
            # Get all versions, ordered by version desc
            with self._session() as session:
                result = session.run(
                    """MATCH (t:Theory {id: $id})
                       RETURN t.version AS version, t.updated_at AS updated_at
                       ORDER BY t.version DESC
                    """,
                    id=tid,
                )
                all_versions = list(result)

            if len(all_versions) <= keep:
                continue

            # Versions to keep (the `keep` newest)
            to_keep = set(r["version"] for r in all_versions[:keep])

            # Delete old versions (not in keep set AND updated_at < cutoff)
            deleted = 0
            for r in all_versions[keep:]:
                if r["version"] in to_keep:
                    continue
                if float(r["updated_at"]) >= cutoff:
                    continue
                with self._session() as session:
                    dr = session.run(
                        """MATCH (t:Theory {id: $id, version: $version})
                           DETACH DELETE t
                           RETURN count(t) AS deleted
                        """,
                        id=tid,
                        version=r["version"],
                    )
                    record = dr.single()
                    if record and record["deleted"] > 0:
                        deleted += 1

            if deleted > 0:
                summary[tid] = deleted

        elapsed = time.time() - t0
        if summary:
            logger.info(
                "Compact older than %d days: %d versions deleted in %.2fs",
                days, sum(summary.values()), elapsed,
            )
        return summary

    # ------------------------------------------------------------------
    # Auto-compact background thread
    # ------------------------------------------------------------------

    def set_auto_compact_enabled(
        self,
        enabled: bool,
        keep_versions: int = 10,
        check_interval_minutes: int = 60,
    ) -> None:
        """Enable or disable periodic auto-compact in a background thread.

        When enabled, a daemon ``threading.Timer`` runs
        ``compact_if_needed`` every *check_interval_minutes* minutes.

        Call ``disable_auto_compact()`` (or ``close()``) to stop.
        """
        with self._auto_compact_lock:
            if enabled == self._auto_compact_enabled:
                return

            if enabled:
                self._auto_compact_enabled = True
                self._auto_compact_keep_versions = max(
                    1, int(keep_versions)
                )
                self._auto_compact_check_interval = (
                    max(1.0, float(check_interval_minutes)) * 60.0
                )
                logger.info(
                    "Auto-compact enabled: keep=%d, interval=%.0fs",
                    self._auto_compact_keep_versions,
                    self._auto_compact_check_interval,
                )
                self._schedule_auto_compact()
            else:
                self._auto_compact_enabled = False
                if self._auto_compact_timer is not None:
                    self._auto_compact_timer.cancel()
                    self._auto_compact_timer = None
                logger.info("Auto-compact disabled")

    def disable_auto_compact(self) -> None:
        """Stop the auto-compact background thread."""
        self.set_auto_compact_enabled(False)

    def _schedule_auto_compact(self) -> None:
        if not self._auto_compact_enabled:
            return
        self._auto_compact_timer = threading.Timer(
            self._auto_compact_check_interval,
            self._run_auto_compact,
        )
        self._auto_compact_timer.daemon = True
        self._auto_compact_timer.start()

    def _run_auto_compact(self) -> None:
        """Run compact_if_needed and reschedule."""
        try:
            self.compact_if_needed(
                keep_versions=self._auto_compact_keep_versions,
                max_versions_before_compact=(
                    self._auto_compact_keep_versions * 2
                ),
            )
        except Exception as exc:
            logger.error("Auto-compact error: %s", exc)
        finally:
            with self._auto_compact_lock:
                if self._auto_compact_enabled:
                    self._schedule_auto_compact()

    # ------------------------------------------------------------------
    # Monitoring / Stats
    # ------------------------------------------------------------------

    def get_version_stats(self, theory_id: str) -> Dict[str, Any]:
        """Return version statistics for a theory.

        Returns:
            A dict with keys:
            - ``total_versions``: Number of versions.
            - ``oldest_version``: Lowest version number.
            - ``newest_version``: Highest version number.
            - ``oldest_updated_at``: Timestamp of the earliest version.
            - ``newest_updated_at``: Timestamp of the latest version.
            - ``estimated_size_bytes``: Rough size estimate (the sum of
              a few bytes per version).
        """
        with self._session() as session:
            result = session.run(
                """MATCH (t:Theory {id: $id})
                   RETURN count(t) AS total,
                          min(t.version) AS oldest,
                          max(t.version) AS newest,
                          min(t.updated_at) AS oldest_ts,
                          max(t.updated_at) AS newest_ts
                """,
                id=theory_id,
            )
            record = result.single()

        if record is None or record["total"] == 0:
            return {
                "total_versions": 0,
                "oldest_version": None,
                "newest_version": None,
                "oldest_updated_at": None,
                "newest_updated_at": None,
                "estimated_size_bytes": 0,
                "theory_id": theory_id,
            }

        total = int(record["total"])
        # Rough estimate: ~2 KB per version node (property storage in Neo4j)
        estimated_bytes = total * 2048

        return {
            "total_versions": total,
            "oldest_version": int(record["oldest"]),
            "newest_version": int(record["newest"]),
            "oldest_updated_at": float(record["oldest_ts"]),
            "newest_updated_at": float(record["newest_ts"]),
            "estimated_size_bytes": estimated_bytes,
            "theory_id": theory_id,
        }

    # ------------------------------------------------------------------
    # Export / Import
    # ------------------------------------------------------------------

    def export_theories(
        self, file_path: str, include_history: bool = True
    ) -> None:
        """Export theories to a JSON file.

        Args:
            file_path: Destination file path.
            include_history: When True (default), all versions are
                exported.  When False, only the latest version of each
                theory is exported.
        """
        if include_history:
            query = (
                "MATCH (t:Theory) RETURN t "
                "ORDER BY t.id, t.version"
            )
        else:
            query = (
                "MATCH (t:Theory) "
                "WHERE NOT EXISTS ((t)-[:NEXT_VERSION]->()) "
                "RETURN t ORDER BY t.created_at DESC"
            )

        with self._session() as session:
            result = session.run(query)
            theories = [
                self._node_to_theory(record["t"]).to_dict()
                for record in result
            ]
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(theories, f, indent=2, ensure_ascii=False)
        logger.info("Exported %d theories to %s", len(theories), file_path)

    def import_theories(
        self,
        file_path: str,
        include_history: bool = True,
        overwrite_existing: bool = False,
    ) -> int:
        """Import theories from a JSON file.

        Args:
            file_path: Source file path.
            include_history: When True (default), all versions in the
                file are imported.  When False, only the latest version
                per unique ``id`` is imported (determined by highest
                ``version`` in the file).
            overwrite_existing: When False (default), existing versions
                are skipped.  When True, conflicting versions are
                overwritten.

        Returns:
            Number of theories imported.
        """
        with open(file_path, "r", encoding="utf-8") as f:
            raw_list = json.load(f)

        if not include_history:
            latest: Dict[str, Dict[str, Any]] = {}
            for raw in raw_list:
                tid = raw.get("id", "")
                ver = int(raw.get("version", 1))
                if tid not in latest or ver > latest[tid].get("version", 0):
                    latest[tid] = raw
            raw_list = list(latest.values())

        count = 0
        for raw in raw_list:
            tid = raw.get("id", "")
            ver = int(raw.get("version", 1))
            if not overwrite_existing:
                existing = self.get_theory_version(tid, ver)
                if existing is not None:
                    continue
            theory = Theory.from_dict(raw)
            self.save_theory(theory)
            count += 1
        logger.info("Imported %d theories from %s", count, file_path)
        return count

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_latest_node(self, theory_id: str) -> Any:
        """Return the raw Neo4j node for the latest version, or None."""
        with self._session() as session:
            result = session.run(
                """MATCH (t:Theory {id: $id})
                   WHERE NOT EXISTS ((t)-[:NEXT_VERSION]->())
                   RETURN t
                """,
                id=theory_id,
            )
            record = result.single()
            return record["t"] if record else None

    @staticmethod
    def _node_to_theory(node: Any) -> Theory:
        """Convert a Neo4j node record to a Theory dataclass."""
        conditions_str = node.get("conditions_json", "{}")
        try:
            conditions = json.loads(conditions_str)
        except (json.JSONDecodeError, TypeError):
            conditions = {}

        return Theory(
            id=node.get("id", ""),
            pattern=node.get("pattern", ""),
            conditions=conditions,
            confidence=float(node.get("confidence", 0.0)),
            provenance=json.loads(node.get("provenance", "[]")),
            version=int(node.get("version", 1)),
            created_at=float(node.get("created_at", 0.0)),
            updated_at=float(node.get("updated_at", 0.0)),
            metadata=json.loads(node.get("metadata", "{}")),
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _safe_cypher_prop(key: str) -> str:
    """Sanitise a condition key for use as a Cypher property name.

    Replaces any character that is not alphanumeric or underscore
    with an underscore, and prepends an underscore if the key starts
    with a digit.
    """
    safe = "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in key)
    if safe and safe[0].isdigit():
        safe = "_" + safe
    return safe if safe else "_cond"
