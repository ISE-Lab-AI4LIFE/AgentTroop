"""Episodic Memory — the living lab notebook of HARMONY-X.

Every interaction between an agent and a victim is recorded as an Episode.
Episodes are stored in SQLite with indexed columns for fast retrieval,
and the full data is kept as a JSON blob for reproducibility.
"""

import hashlib
import json
import sqlite3
import subprocess
import time
import uuid
from array import array
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple

from core.types import Outcome

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_GIT_HASH_CACHE: Optional[str] = None


def _get_git_hash() -> str:
    global _GIT_HASH_CACHE
    if _GIT_HASH_CACHE is not None:
        return _GIT_HASH_CACHE
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            _GIT_HASH_CACHE = result.stdout.strip()
            return _GIT_HASH_CACHE
    except Exception:
        pass
    try:
        git_head = Path(".git/HEAD")
        if git_head.exists():
            _GIT_HASH_CACHE = git_head.read_text().strip()
            return _GIT_HASH_CACHE
    except Exception:
        pass
    _GIT_HASH_CACHE = "unknown"
    return _GIT_HASH_CACHE


def _compute_checksum(data: Dict[str, Any]) -> str:
    canonical = json.dumps(data, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Data classes — transformation trace
# ---------------------------------------------------------------------------


@dataclass
class TransformStep:
    """A single transformation step with input and output."""
    transform_name: str
    parameters: Dict[str, Any]
    input_prompt: str
    output_prompt: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "transform_name": self.transform_name,
            "parameters": self.parameters,
            "input_prompt": self.input_prompt,
            "output_prompt": self.output_prompt,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TransformStep":
        return cls(
            transform_name=data["transform_name"],
            parameters=dict(data.get("parameters", {})),
            input_prompt=data.get("input_prompt", ""),
            output_prompt=data.get("output_prompt", ""),
        )


@dataclass
class TransformationTrace:
    """Complete trace of all transformations applied to a prompt."""
    steps: List[TransformStep] = field(default_factory=list)

    @property
    def original_prompt(self) -> str:
        return self.steps[0].input_prompt if self.steps else ""

    @property
    def final_prompt(self) -> str:
        return self.steps[-1].output_prompt if self.steps else ""

    def to_dict(self) -> Dict[str, Any]:
        return {"steps": [s.to_dict() for s in self.steps]}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TransformationTrace":
        steps = [TransformStep.from_dict(s) for s in data.get("steps", [])]
        return cls(steps=steps)


# ---------------------------------------------------------------------------
# Data classes — provenance
# ---------------------------------------------------------------------------


@dataclass
class Provenance:
    """Standardised provenance metadata for reproducibility."""
    code_version: str = "unknown"
    agent_version: str = ""
    experiment_config_hash: str = ""
    dataset_version: str = ""
    environment_id: str = ""

    def __post_init__(self) -> None:
        if not self.code_version or self.code_version == "unknown":
            self.code_version = _get_git_hash()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "code_version": self.code_version,
            "agent_version": self.agent_version,
            "experiment_config_hash": self.experiment_config_hash,
            "dataset_version": self.dataset_version,
            "environment_id": self.environment_id,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Provenance":
        return cls(
            code_version=data.get("code_version", "unknown"),
            agent_version=data.get("agent_version", ""),
            experiment_config_hash=data.get("experiment_config_hash", ""),
            dataset_version=data.get("dataset_version", ""),
            environment_id=data.get("environment_id", ""),
        )


# ---------------------------------------------------------------------------
# Data classes — core records
# ---------------------------------------------------------------------------


@dataclass
class InterventionRecord:
    """A recorded intervention sent to a victim.

    Stores both the flat ``transforms`` list (for lightweight use)
    and an optional ``transformation_trace`` (for detailed per-step
    input/output tracking). When both are present the trace is
    authoritative; the flat fields are kept for backward compatibility.
    """

    intervention_id: str
    prompt: str
    transforms: List[Dict[str, Any]] = field(default_factory=list)
    final_prompt: str = ""
    transformation_trace: Optional[TransformationTrace] = None
    strategy_name: str = ""
    agent_name: str = ""
    hypothesis_id: Optional[str] = None
    session_id: str = ""
    parent_intervention_id: Optional[str] = None
    iteration: int = 0
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "intervention_id": self.intervention_id,
            "prompt": self.prompt,
            "transforms": self.transforms,
            "final_prompt": self.final_prompt,
            "strategy_name": self.strategy_name,
            "agent_name": self.agent_name,
            "hypothesis_id": self.hypothesis_id,
            "session_id": self.session_id,
            "parent_intervention_id": self.parent_intervention_id,
            "iteration": self.iteration,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
        }
        if self.transformation_trace is not None:
            d["transformation_trace"] = self.transformation_trace.to_dict()
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "InterventionRecord":
        trace_data = data.get("transformation_trace")
        trace = (
            TransformationTrace.from_dict(trace_data)
            if trace_data
            else None
        )
        return cls(
            intervention_id=data.get("intervention_id", ""),
            prompt=data.get("prompt", ""),
            transforms=list(data.get("transforms", [])),
            final_prompt=data.get("final_prompt", ""),
            transformation_trace=trace,
            strategy_name=data.get("strategy_name", ""),
            agent_name=data.get("agent_name", ""),
            hypothesis_id=data.get("hypothesis_id"),
            session_id=data.get("session_id", ""),
            parent_intervention_id=data.get("parent_intervention_id"),
            iteration=int(data.get("iteration", 0)),
            timestamp=float(data.get("timestamp", time.time())),
            metadata=dict(data.get("metadata", {})),
        )


@dataclass
class Episode:
    """An atomic evidence unit: one intervention + its outcome.

    ``hypothesis_support`` is deliberately **not** stored here.
    Episode-hypothesis relationships live in the ``EpisodeEvidence``
    table, which is populated during analysis — not during collection.
    """

    episode_id: str
    intervention: InterventionRecord
    victim_name: str
    campaign_id: str
    experiment_id: str
    outcome: Outcome
    session_id: str = ""
    parent_episode_id: Optional[str] = None
    raw_response: str = ""
    latency_ms: float = 0.0
    token_usage: Optional[Dict[str, Any]] = None
    checksum: str = ""
    annotations: Dict[str, Any] = field(default_factory=dict)
    provenance: Provenance = field(default_factory=Provenance)
    is_archived: bool = False
    created_at: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.episode_id:
            self.episode_id = f"ep_{uuid.uuid4().hex[:12]}"
        if not self.checksum:
            self.checksum = self._compute_own_checksum()

    def _data_for_checksum(self) -> Dict[str, Any]:
        d = self.to_dict()
        d.pop("checksum", None)
        return d

    def _compute_own_checksum(self) -> str:
        return _compute_checksum(self._data_for_checksum())

    def verify_checksum(self) -> bool:
        return self.checksum == self._compute_own_checksum()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "intervention": self.intervention.to_dict(),
            "victim_name": self.victim_name,
            "campaign_id": self.campaign_id,
            "experiment_id": self.experiment_id,
            "session_id": self.session_id,
            "parent_episode_id": self.parent_episode_id,
            "outcome": int(self.outcome),
            "raw_response": self.raw_response,
            "latency_ms": self.latency_ms,
            "token_usage": self.token_usage,
            "checksum": self.checksum,
            "annotations": self.annotations,
            "provenance": self.provenance.to_dict(),
            "is_archived": self.is_archived,
            "created_at": self.created_at,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Episode":
        intervention = InterventionRecord.from_dict(
            data.get("intervention", {})
        )
        prov_data = data.get("provenance", {})
        return cls(
            episode_id=data.get("episode_id", ""),
            intervention=intervention,
            victim_name=data.get("victim_name", ""),
            campaign_id=data.get("campaign_id", ""),
            experiment_id=data.get("experiment_id", ""),
            session_id=data.get("session_id", ""),
            parent_episode_id=data.get("parent_episode_id"),
            outcome=int(data.get("outcome", 0)),
            raw_response=data.get("raw_response", ""),
            latency_ms=float(data.get("latency_ms", 0.0)),
            token_usage=data.get("token_usage"),
            checksum=data.get("checksum", ""),
            annotations=dict(data.get("annotations", {})),
            provenance=Provenance.from_dict(prov_data),
            is_archived=bool(data.get("is_archived", False)),
            created_at=float(data.get("created_at", time.time())),
            metadata=dict(data.get("metadata", {})),
        )


@dataclass
class EpisodeEvidence:
    """A link between an episode and a hypothesis.

    Populated during analysis — not during collection.
    The ``relationship`` field is one of ``'supporting'``,
    ``'contradicting'``, or ``'neutral'``.
    """

    evidence_id: str = ""
    episode_id: str = ""
    hypothesis_id: str = ""
    relationship: str = "neutral"
    strength: float = 1.0
    created_at: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.evidence_id:
            self.evidence_id = f"ev_{uuid.uuid4().hex[:12]}"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "evidence_id": self.evidence_id,
            "episode_id": self.episode_id,
            "hypothesis_id": self.hypothesis_id,
            "relationship": self.relationship,
            "strength": self.strength,
            "created_at": self.created_at,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EpisodeEvidence":
        return cls(
            evidence_id=data.get("evidence_id", ""),
            episode_id=data["episode_id"],
            hypothesis_id=data["hypothesis_id"],
            relationship=data.get("relationship", "neutral"),
            strength=float(data.get("strength", 1.0)),
            created_at=float(data.get("created_at", time.time())),
            metadata=dict(data.get("metadata", {})),
        )


@dataclass
class EpisodeFilter:
    """Composite filter criteria for querying episodes.

    By default, archived (soft-deleted) episodes are **excluded**.
    Set ``include_archived=True`` to include them.
    Set ``only_archived=True`` to see only archived ones.
    """

    campaign_id: Optional[str] = None
    experiment_id: Optional[str] = None
    session_id: Optional[str] = None
    victim_name: Optional[str] = None
    outcome: Optional[Outcome] = None
    agent_name: Optional[str] = None
    strategy_name: Optional[str] = None
    hypothesis_id: Optional[str] = None
    parent_episode_id: Optional[str] = None
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    tags: Optional[Dict[str, Any]] = None
    include_archived: bool = False
    only_archived: bool = False


# ---------------------------------------------------------------------------
# EpisodicMemory
# ---------------------------------------------------------------------------


class EpisodicMemory:
    """Persistent store for experimental episodes backed by SQLite.

    Supports up to 100 000 episodes with sub-100ms indexed lookups.
    """

    def __init__(self, db_path: str = "episodic_memory.db") -> None:
        self.db_path = db_path
        parent = Path(db_path).parent
        if str(parent) != "." and not parent.exists():
            parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(db_path)
        self._conn.row_factory = sqlite3.Row
        self._init_schema()

    def close(self) -> None:
        self._conn.close()

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------

    def _init_schema(self) -> None:
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS episodes (
                episode_id        TEXT PRIMARY KEY,
                campaign_id       TEXT NOT NULL,
                experiment_id     TEXT NOT NULL,
                session_id        TEXT NOT NULL DEFAULT '',
                victim_name       TEXT NOT NULL,
                outcome           INTEGER NOT NULL,
                strategy_name     TEXT NOT NULL DEFAULT '',
                agent_name        TEXT NOT NULL DEFAULT '',
                iteration         INTEGER NOT NULL DEFAULT 0,
                hypothesis_id     TEXT NOT NULL DEFAULT '',
                parent_episode_id TEXT,
                checksum          TEXT NOT NULL,
                is_archived       INTEGER NOT NULL DEFAULT 0,
                timestamp         REAL NOT NULL,
                created_at        REAL NOT NULL,
                data              TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_ep_campaign
                ON episodes(campaign_id);
            CREATE INDEX IF NOT EXISTS idx_ep_experiment
                ON episodes(experiment_id);
            CREATE INDEX IF NOT EXISTS idx_ep_session
                ON episodes(session_id);
            CREATE INDEX IF NOT EXISTS idx_ep_victim
                ON episodes(victim_name);
            CREATE INDEX IF NOT EXISTS idx_ep_outcome
                ON episodes(outcome);
            CREATE INDEX IF NOT EXISTS idx_ep_strategy
                ON episodes(strategy_name);
            CREATE INDEX IF NOT EXISTS idx_ep_agent
                ON episodes(agent_name);

            CREATE INDEX IF NOT EXISTS idx_ep_hypothesis
                ON episodes(hypothesis_id);
            CREATE INDEX IF NOT EXISTS idx_ep_parent
                ON episodes(parent_episode_id);
            CREATE INDEX IF NOT EXISTS idx_ep_archived
                ON episodes(is_archived);
            CREATE INDEX IF NOT EXISTS idx_ep_timestamp
                ON episodes(timestamp);

            CREATE TABLE IF NOT EXISTS episode_evidence (
                evidence_id    TEXT PRIMARY KEY,
                episode_id     TEXT NOT NULL,
                hypothesis_id  TEXT NOT NULL,
                relationship   TEXT NOT NULL,
                strength       REAL NOT NULL DEFAULT 1.0,
                created_at     REAL NOT NULL,
                metadata       TEXT NOT NULL DEFAULT '{}',
                FOREIGN KEY (episode_id) REFERENCES episodes(episode_id)
            );

            CREATE INDEX IF NOT EXISTS idx_ev_episode
                ON episode_evidence(episode_id);
            CREATE INDEX IF NOT EXISTS idx_ev_hypothesis
                ON episode_evidence(hypothesis_id);
        """)
        self._conn.commit()

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    def save_episode(self, episode: Episode) -> str:
        """Persist an episode. Returns its ``episode_id``.

        Upsert semantics — if an episode with the same ID exists it is
        replaced. The checksum is recomputed automatically.
        """
        episode.checksum = episode._compute_own_checksum()
        data_json = json.dumps(episode.to_dict(), ensure_ascii=False)
        ts = episode.intervention.timestamp or episode.created_at
        self._conn.execute(
            """INSERT OR REPLACE INTO episodes
               (episode_id, campaign_id, experiment_id, session_id,
                victim_name, outcome, strategy_name, agent_name,
                iteration, hypothesis_id, parent_episode_id, checksum,
                is_archived, timestamp, created_at, data)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                episode.episode_id,
                episode.campaign_id,
                episode.experiment_id,
                episode.session_id,
                episode.victim_name,
                int(episode.outcome),
                episode.intervention.strategy_name,
                episode.intervention.agent_name,
                episode.intervention.iteration,
                episode.intervention.hypothesis_id or "",
                episode.parent_episode_id,
                episode.checksum,
                1 if episode.is_archived else 0,
                ts,
                episode.created_at,
                data_json,
            ),
        )
        self._conn.commit()
        return episode.episode_id

    def get_episode(self, episode_id: str) -> Optional[Episode]:
        """Retrieve a single episode by ID, or None if not found."""
        row = self._conn.execute(
            "SELECT data FROM episodes WHERE episode_id = ?",
            (episode_id,),
        ).fetchone()
        if row is None:
            return None
        return Episode.from_dict(json.loads(row["data"]))

    def episode_exists(self, episode_id: str) -> bool:
        row = self._conn.execute(
            "SELECT 1 FROM episodes WHERE episode_id = ?",
            (episode_id,),
        ).fetchone()
        return row is not None

    def delete_episode(self, episode_id: str) -> bool:
        """Soft-delete an episode (set ``is_archived = 1``).

        Use ``hard_delete_episode`` to remove permanently.
        Returns True if a row was updated.
        """
        cursor = self._conn.execute(
            "UPDATE episodes SET is_archived = 1 WHERE episode_id = ?",
            (episode_id,),
        )
        self._conn.commit()
        return cursor.rowcount > 0

    def hard_delete_episode(self, episode_id: str) -> bool:
        """Permanently remove an episode and its evidence."""
        self._conn.execute(
            "DELETE FROM episode_evidence WHERE episode_id = ?",
            (episode_id,),
        )
        cursor = self._conn.execute(
            "DELETE FROM episodes WHERE episode_id = ?",
            (episode_id,),
        )
        self._conn.commit()
        return cursor.rowcount > 0

    def restore_episode(self, episode_id: str) -> bool:
        """Un-archive a previously soft-deleted episode."""
        cursor = self._conn.execute(
            "UPDATE episodes SET is_archived = 0 WHERE episode_id = ?",
            (episode_id,),
        )
        self._conn.commit()
        return cursor.rowcount > 0

    def delete_campaign(self, campaign_id: str) -> int:
        """Soft-delete all episodes in a campaign. Returns count."""
        cursor = self._conn.execute(
            "UPDATE episodes SET is_archived = 1 WHERE campaign_id = ?",
            (campaign_id,),
        )
        self._conn.commit()
        return cursor.rowcount

    def hard_delete_campaign(self, campaign_id: str) -> int:
        """Permanently remove all episodes of a campaign."""
        self._conn.execute(
            "DELETE FROM episode_evidence WHERE episode_id IN "
            "(SELECT episode_id FROM episodes WHERE campaign_id = ?)",
            (campaign_id,),
        )
        cursor = self._conn.execute(
            "DELETE FROM episodes WHERE campaign_id = ?",
            (campaign_id,),
        )
        self._conn.commit()
        return cursor.rowcount

    # ------------------------------------------------------------------
    # Query helpers
    # ------------------------------------------------------------------

    def _build_where_clause(
        self, filter: EpisodeFilter
    ) -> Tuple[str, List[Any]]:
        clauses: List[str] = ["1=1"]
        params: List[Any] = []

        if filter.campaign_id is not None:
            clauses.append("campaign_id = ?")
            params.append(filter.campaign_id)
        if filter.experiment_id is not None:
            clauses.append("experiment_id = ?")
            params.append(filter.experiment_id)
        if filter.session_id is not None:
            clauses.append("session_id = ?")
            params.append(filter.session_id)
        if filter.victim_name is not None:
            clauses.append("victim_name = ?")
            params.append(filter.victim_name)
        if filter.outcome is not None:
            clauses.append("outcome = ?")
            params.append(int(filter.outcome))
        if filter.agent_name is not None:
            clauses.append("agent_name = ?")
            params.append(filter.agent_name)
        if filter.strategy_name is not None:
            clauses.append("strategy_name = ?")
            params.append(filter.strategy_name)
        if filter.parent_episode_id is not None:
            clauses.append("parent_episode_id = ?")
            params.append(filter.parent_episode_id)
        if filter.start_time is not None:
            clauses.append("timestamp >= ?")
            params.append(filter.start_time)
        if filter.end_time is not None:
            clauses.append("timestamp <= ?")
            params.append(filter.end_time)
        if filter.only_archived:
            clauses.append("is_archived = 1")
        elif not filter.include_archived:
            clauses.append("is_archived = 0")

        if filter.hypothesis_id is not None:
            h = filter.hypothesis_id
            clauses.append(
                "(hypothesis_id = ? OR "
                "episode_id IN ("
                "SELECT episode_id FROM episode_evidence "
                "WHERE hypothesis_id = ?))"
            )
            params.extend([h, h])

        where = " AND ".join(clauses)
        return where, params

    def _filter_data_side(
        self, episodes: List[Episode], filter: EpisodeFilter
    ) -> List[Episode]:
        if filter.tags is not None:
            episodes = [
                e
                for e in episodes
                if all(
                    e.annotations.get(k) == v
                    for k, v in filter.tags.items()
                )
            ]
        return episodes

    # ------------------------------------------------------------------
    # Query methods
    # ------------------------------------------------------------------

    def get_episodes_by_campaign(
        self, campaign_id: str
    ) -> List[Episode]:
        return self._fetch_where(
            "campaign_id = ? AND is_archived = 0", [campaign_id]
        )

    def get_episodes_by_experiment(
        self, experiment_id: str
    ) -> List[Episode]:
        return self._fetch_where(
            "experiment_id = ? AND is_archived = 0", [experiment_id]
        )

    def get_episodes_by_session(
        self, session_id: str
    ) -> List[Episode]:
        return self._fetch_where(
            "session_id = ? AND is_archived = 0", [session_id]
        )

    def get_episodes_by_timerange(
        self, start: float, end: float
    ) -> List[Episode]:
        return self._fetch_where(
            "timestamp >= ? AND timestamp <= ? AND is_archived = 0",
            [start, end],
        )

    def get_children_episodes(
        self, parent_episode_id: str
    ) -> List[Episode]:
        """Return all episodes that have the given parent."""
        return self._fetch_where(
            "parent_episode_id = ? AND is_archived = 0",
            [parent_episode_id],
        )

    def _fetch_where(
        self, where_clause: str, params: List[Any]
    ) -> List[Episode]:
        cursor = self._conn.execute(
            f"SELECT data FROM episodes WHERE {where_clause} "
            "ORDER BY timestamp",
            params,
        )
        return [
            Episode.from_dict(json.loads(row["data"])) for row in cursor
        ]

    def filter_episodes(
        self, filter: EpisodeFilter
    ) -> List[Episode]:
        where_clause, params = self._build_where_clause(filter)
        cursor = self._conn.execute(
            f"SELECT data FROM episodes WHERE {where_clause} "
            "ORDER BY timestamp",
            params,
        )
        episodes = [
            Episode.from_dict(json.loads(row["data"])) for row in cursor
        ]
        return self._filter_data_side(episodes, filter)

    # ------------------------------------------------------------------
    # EpisodeEvidence
    # ------------------------------------------------------------------

    def add_evidence(self, evidence: EpisodeEvidence) -> str:
        self._conn.execute(
            """INSERT OR REPLACE INTO episode_evidence
               (evidence_id, episode_id, hypothesis_id, relationship,
                strength, created_at, metadata)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                evidence.evidence_id,
                evidence.episode_id,
                evidence.hypothesis_id,
                evidence.relationship,
                evidence.strength,
                evidence.created_at,
                json.dumps(evidence.metadata, ensure_ascii=False),
            ),
        )
        self._conn.commit()
        return evidence.evidence_id

    def get_evidence_for_hypothesis(
        self, hypothesis_id: str
    ) -> List[EpisodeEvidence]:
        cursor = self._conn.execute(
            "SELECT * FROM episode_evidence WHERE hypothesis_id = ?",
            (hypothesis_id,),
        )
        return [_row_to_evidence(row) for row in cursor]

    def get_evidence_for_episode(
        self, episode_id: str
    ) -> List[EpisodeEvidence]:
        cursor = self._conn.execute(
            "SELECT * FROM episode_evidence WHERE episode_id = ?",
            (episode_id,),
        )
        return [_row_to_evidence(row) for row in cursor]

    def delete_evidence(self, evidence_id: str) -> bool:
        cursor = self._conn.execute(
            "DELETE FROM episode_evidence WHERE evidence_id = ?",
            (evidence_id,),
        )
        self._conn.commit()
        return cursor.rowcount > 0

    # ------------------------------------------------------------------
    # Annotations
    # ------------------------------------------------------------------

    def add_annotation(
        self, episode_id: str, key: str, value: Any
    ) -> None:
        row = self._conn.execute(
            "SELECT data FROM episodes WHERE episode_id = ?",
            (episode_id,),
        ).fetchone()
        if row is None:
            raise KeyError(f"Episode '{episode_id}' not found")
        data = json.loads(row["data"])
        data.setdefault("annotations", {})[key] = value
        self._conn.execute(
            "UPDATE episodes SET data = ? WHERE episode_id = ?",
            (json.dumps(data, ensure_ascii=False), episode_id),
        )
        self._conn.commit()

    def get_episodes_with_annotation(
        self, key: str, value: Any = None
    ) -> List[Episode]:
        all_rows = self._conn.execute(
            "SELECT data FROM episodes WHERE is_archived = 0 "
            "ORDER BY timestamp"
        ).fetchall()
        result: List[Episode] = []
        for row in all_rows:
            ep = Episode.from_dict(json.loads(row["data"]))
            if key in ep.annotations:
                if value is None or ep.annotations[key] == value:
                    result.append(ep)
        return result

    # ------------------------------------------------------------------
    # Checksum verification
    # ------------------------------------------------------------------

    def verify_episode_checksum(self, episode_id: str) -> bool:
        ep = self.get_episode(episode_id)
        if ep is None:
            return False
        return ep.verify_checksum()

    def find_corrupted_episodes(
        self, campaign_id: Optional[str] = None
    ) -> List[str]:
        if campaign_id:
            cursor = self._conn.execute(
                "SELECT episode_id, data FROM episodes "
                "WHERE campaign_id = ?",
                (campaign_id,),
            )
        else:
            cursor = self._conn.execute(
                "SELECT episode_id, data FROM episodes"
            )
        corrupted: List[str] = []
        for row in cursor:
            ep = Episode.from_dict(json.loads(row["data"]))
            if not ep.verify_checksum():
                corrupted.append(row["episode_id"])
        return corrupted

    # ------------------------------------------------------------------
    # Export / Import / Snapshot
    # ------------------------------------------------------------------

    def export_campaign(
        self, campaign_id: str, output_dir: str
    ) -> str:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        path = out / f"{campaign_id}.jsonl"
        episodes = self._fetch_where(
            "campaign_id = ? AND is_archived = 0", [campaign_id]
        )
        with open(path, "w") as f:
            for ep in episodes:
                f.write(json.dumps(ep.to_dict(), ensure_ascii=False) + "\n")
        return str(path)

    def import_campaign(self, input_path: str) -> int:
        count = 0
        with open(input_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                ep = Episode.from_dict(json.loads(line))
                self.save_episode(ep)
                count += 1
        return count

    def snapshot_campaign(
        self,
        campaign_id: str,
        output_path: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Export a complete campaign as a single JSON artifact.

        Includes all episodes, evidence, and optional metadata/config
        for full reproducibility.
        """
        episodes = self._fetch_where(
            "campaign_id = ? AND is_archived = 0", [campaign_id]
        )
        evidence = self._conn.execute(
            "SELECT * FROM episode_evidence WHERE episode_id IN "
            "(SELECT episode_id FROM episodes WHERE campaign_id = ?)",
            (campaign_id,),
        ).fetchall()
        snapshot = {
            "snapshot_version": "1.0",
            "campaign_id": campaign_id,
            "created_at": time.time(),
            "metadata": metadata or {},
            "episode_count": len(episodes),
            "evidence_count": len(evidence),
            "episodes": [ep.to_dict() for ep in episodes],
            "evidence": [_row_to_evidence(r).to_dict() for r in evidence],
        }
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            json.dump(snapshot, f, indent=2, ensure_ascii=False)
        return str(out)

    # ------------------------------------------------------------------
    # Reproducibility
    # ------------------------------------------------------------------

    def reconstruct_campaign(
        self, campaign_id: str, include_archived: bool = False
    ) -> Generator[Episode, None, None]:
        archived_clause = "" if include_archived else "AND is_archived = 0"
        cursor = self._conn.execute(
            f"SELECT data FROM episodes WHERE campaign_id = ? "
            f"{archived_clause} ORDER BY timestamp ASC",
            (campaign_id,),
        )
        for row in cursor:
            yield Episode.from_dict(json.loads(row["data"]))


def _row_to_evidence(row: sqlite3.Row) -> EpisodeEvidence:
    return EpisodeEvidence(
        evidence_id=row["evidence_id"],
        episode_id=row["episode_id"],
        hypothesis_id=row["hypothesis_id"],
        relationship=row["relationship"],
        strength=float(row["strength"]),
        created_at=float(row["created_at"]),
        metadata=json.loads(row["metadata"]) if row["metadata"] else {},
    )
