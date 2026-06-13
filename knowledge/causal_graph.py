from __future__ import annotations

import json
import logging
import re
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from neo4j import GraphDatabase, Session, basic_auth
from neo4j.exceptions import ServiceUnavailable

try:
    from scipy.stats import fisher_exact
except ImportError:
    fisher_exact = None

try:
    from scipy.stats import bayes_mvs
    _HAS_BAYES = True
except ImportError:
    _HAS_BAYES = False

logger = logging.getLogger(__name__)

_MIN_INTERVENTIONS_FOR_EDGE = 10  # minimum trials per do-condition
# Minimum total observations across both conditions for statistical significance.
_MIN_TOTAL_OBSERVATIONS = 20
_P_VALUE_THRESHOLD = 0.05

_CYPHER_PROP_RE = re.compile(r"[^a-zA-Z0-9_]")


def _safe_prop(key: str) -> str:
    safe = _CYPHER_PROP_RE.sub("_", key)
    if safe and safe[0].isdigit():
        safe = "_" + safe
    return safe if safe else "_prop"


@dataclass
class CausalNode:
    id: str = ""
    name: str = ""
    type: str = "primitive"
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)

    def __post_init__(self) -> None:
        if not self.id:
            self.id = f"cnd_{uuid.uuid4().hex[:12]}"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "type": self.type,
            "metadata": self.metadata,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CausalNode":
        return cls(
            id=data.get("id", ""),
            name=data.get("name", ""),
            type=data.get("type", "primitive"),
            metadata=dict(data.get("metadata", {})),
            created_at=float(data.get("created_at", time.time())),
        )


@dataclass
class CausalEdge:
    source_id: str
    target_id: str
    strength: float = 0.0
    p_value: float = 1.0
    intervention_ids: List[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)

    def __post_init__(self) -> None:
        self.strength = max(0.0, min(1.0, float(self.strength)))
        self.p_value = max(0.0, min(1.0, float(self.p_value)))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_id": self.source_id,
            "target_id": self.target_id,
            "strength": self.strength,
            "p_value": self.p_value,
            "intervention_ids": self.intervention_ids,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CausalEdge":
        return cls(
            source_id=data["source_id"],
            target_id=data["target_id"],
            strength=float(data.get("strength", 0.0)),
            p_value=float(data.get("p_value", 1.0)),
            intervention_ids=list(data.get("intervention_ids", [])),
            created_at=float(data.get("created_at", time.time())),
        )


class CausalGraph:
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
        self._cumulative_outcomes: Dict[str, Dict[str, List[int]]] = {}
        self._ensure_constraints()

    def _cum_key(self, source_name: str, target_name: str) -> str:
        return f"{source_name}||{target_name}"

    def _get_driver(self) -> Any:
        if self._driver is None:
            try:
                self._driver = GraphDatabase.driver(
                    self.uri, auth=basic_auth(self.user, self.password)
                )
            except ServiceUnavailable as exc:
                raise ConnectionError(
                    f"Cannot connect to Neo4j at {self.uri}: {exc}"
                ) from exc
        return self._driver

    def _session(self) -> Session:
        return self._get_driver().session(database=self.database)

    def close(self) -> None:
        if self._driver is not None:
            self._driver.close()
            self._driver = None

    def __enter__(self) -> "CausalGraph":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    def _ensure_constraints(self) -> None:
        with self._session() as session:
            session.run(
                "CREATE CONSTRAINT IF NOT EXISTS "
                "FOR (n:CausalNode) REQUIRE n.id IS UNIQUE"
            )
            session.run(
                "CREATE INDEX IF NOT EXISTS FOR (n:CausalNode) "
                "ON (n.type)"
            )
            session.run(
                "CREATE INDEX IF NOT EXISTS FOR (n:CausalNode) "
                "ON (n.name)"
            )
            session.run(
                "CREATE CONSTRAINT IF NOT EXISTS "
                "FOR ()-[r:CAUSAL_EDGE]-() "
                "REQUIRE (r.source_id, r.target_id) IS UNIQUE"
            )

    def add_node(
        self,
        name: str,
        node_type: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        node = CausalNode(
            name=name,
            type=node_type,
            metadata=metadata or {},
        )
        with self._session() as session:
            session.run(
                """CREATE (n:CausalNode {
                    id: $id,
                    name: $name,
                    type: $type,
                    metadata: $metadata,
                    created_at: $created_at
                })""",
                id=node.id,
                name=node.name,
                type=node.type,
                metadata=json.dumps(node.metadata, ensure_ascii=False),
                created_at=node.created_at,
            )
        logger.debug("Added causal node %s (%s)", node.id, name)
        return node.id

    def get_node(self, node_id: str) -> Optional[CausalNode]:
        with self._session() as session:
            result = session.run(
                "MATCH (n:CausalNode {id: $id}) RETURN n",
                id=node_id,
            )
            record = result.single()
            if record is None:
                return None
            return self._node_to_causal_node(record["n"])

    def get_or_create_node(
        self,
        name: str,
        node_type: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        existing = self.find_nodes_by_name(name)
        if existing:
            return existing[0].id
        return self.add_node(name, node_type, metadata)

    def find_nodes_by_name(self, name: str) -> List[CausalNode]:
        with self._session() as session:
            result = session.run(
                "MATCH (n:CausalNode {name: $name}) RETURN n ORDER BY n.created_at",
                name=name,
            )
            return [self._node_to_causal_node(rec["n"]) for rec in result]

    def find_nodes_by_type(self, node_type: str) -> List[CausalNode]:
        with self._session() as session:
            result = session.run(
                "MATCH (n:CausalNode {type: $type}) RETURN n ORDER BY n.name",
                type=node_type,
            )
            return [self._node_to_causal_node(rec["n"]) for rec in result]

    def update_node(
        self,
        node_id: str,
        name: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        sets: List[str] = []
        params: Dict[str, Any] = {"id": node_id}
        if name is not None:
            sets.append("n.name = $name")
            params["name"] = name
        if metadata is not None:
            sets.append("n.metadata = $metadata")
            params["metadata"] = json.dumps(metadata, ensure_ascii=False)
        if not sets:
            return False
        set_clause = ", ".join(sets)
        with self._session() as session:
            result = session.run(
                f"MATCH (n:CausalNode {{id: $id}}) SET {set_clause} "
                "RETURN count(n) AS updated",
                **params,
            )
            record = result.single()
            return record is not None and record["updated"] > 0

    def delete_node(self, node_id: str) -> bool:
        with self._session() as session:
            tx = session.begin_transaction()
            try:
                tx.run(
                    "MATCH (n:CausalNode {id: $id}) "
                    "OPTIONAL MATCH (n)-[r:CAUSAL_EDGE]-() "
                    "DELETE r",
                    id=node_id,
                )
                result = tx.run(
                    "MATCH (n:CausalNode {id: $id}) "
                    "DELETE n "
                    "RETURN count(n) AS deleted",
                    id=node_id,
                )
                record = result.single()
                tx.commit()
            except Exception:
                tx.rollback()
                raise
            return record is not None and record["deleted"] > 0

    def add_edge(
        self,
        source_id: str,
        target_id: str,
        strength: float,
        p_value: float,
        intervention_ids: List[str],
        num_trials: int = 0,
    ) -> bool:
        if p_value > _P_VALUE_THRESHOLD:
            raise ValueError(
                f"Causal edge requires p-value <= {_P_VALUE_THRESHOLD}, "
                f"got {p_value:.4f}. Use compute_p_value() to re-evaluate."
            )
        strength = max(0.0, min(1.0, float(strength)))
        p_value = max(0.0, min(1.0, float(p_value)))
        edge = CausalEdge(
            source_id=source_id,
            target_id=target_id,
            strength=strength,
            p_value=p_value,
            intervention_ids=intervention_ids,
        )
        with self._session() as session:
            result = session.run(
                """MATCH (s:CausalNode {id: $sid})
                   MATCH (t:CausalNode {id: $tid})
                   MERGE (s)-[r:CAUSAL_EDGE {source_id: $sid, target_id: $tid}]->
                         (t)
                   SET r.strength = $strength,
                       r.p_value = $p_value,
                       r.intervention_ids = $iids,
                       r.created_at = $created_at
                   RETURN count(r) AS created
                """,
                sid=edge.source_id,
                tid=edge.target_id,
                strength=edge.strength,
                p_value=edge.p_value,
                iids=json.dumps(edge.intervention_ids, ensure_ascii=False),
                created_at=edge.created_at,
            )
            record = result.single()
            return record is not None and record["created"] > 0

    def get_edge(
        self, source_id: str, target_id: str
    ) -> Optional[CausalEdge]:
        with self._session() as session:
            result = session.run(
                "MATCH (s:CausalNode {id: $sid})"
                "-[r:CAUSAL_EDGE {source_id: $sid, target_id: $tid}]-"
                "(t:CausalNode {id: $tid}) "
                "RETURN r",
                sid=source_id,
                tid=target_id,
            )
            record = result.single()
            if record is None:
                return None
            return self._edge_to_causal_edge(record["r"])

    def delete_edge(self, source_id: str, target_id: str) -> bool:
        with self._session() as session:
            result = session.run(
                "MATCH (s:CausalNode {id: $sid})"
                "-[r:CAUSAL_EDGE {source_id: $sid, target_id: $tid}]-"
                "(t:CausalNode {id: $tid}) "
                "DELETE r "
                "RETURN count(r) AS deleted",
                sid=source_id,
                tid=target_id,
            )
            record = result.single()
            return record is not None and record["deleted"] > 0

    def find_causes(
        self, target_id: str, min_strength: float = 0.0
    ) -> List[Tuple[CausalEdge, CausalNode]]:
        with self._session() as session:
            result = session.run(
                """MATCH (source:CausalNode)-[r:CAUSAL_EDGE]->(target:CausalNode {id: $tid})
                   WHERE r.strength >= $min_str
                   RETURN r, source
                   ORDER BY r.strength DESC
                """,
                tid=target_id,
                min_str=min_strength,
            )
            items: List[Tuple[CausalEdge, CausalNode]] = []
            for rec in result:
                edge = self._edge_to_causal_edge(rec["r"])
                node = self._node_to_causal_node(rec["source"])
                items.append((edge, node))
            return items

    def find_effects(
        self, source_id: str, min_strength: float = 0.0
    ) -> List[Tuple[CausalEdge, CausalNode]]:
        with self._session() as session:
            result = session.run(
                """MATCH (source:CausalNode {id: $sid})-[r:CAUSAL_EDGE]->(target:CausalNode)
                   WHERE r.strength >= $min_str
                   RETURN r, target
                   ORDER BY r.strength DESC
                """,
                sid=source_id,
                min_str=min_strength,
            )
            items: List[Tuple[CausalEdge, CausalNode]] = []
            for rec in result:
                edge = self._edge_to_causal_edge(rec["r"])
                node = self._node_to_causal_node(rec["target"])
                items.append((edge, node))
            return items

    def get_all_nodes(self) -> List[CausalNode]:
        with self._session() as session:
            result = session.run(
                "MATCH (n:CausalNode) RETURN n ORDER BY n.name"
            )
            return [self._node_to_causal_node(rec["n"]) for rec in result]

    def get_all_edges(self) -> List[CausalEdge]:
        with self._session() as session:
            result = session.run(
                "MATCH ()-[r:CAUSAL_EDGE]->() "
                "RETURN r ORDER BY r.strength DESC"
            )
            return [self._edge_to_causal_edge(rec["r"]) for rec in result]

    def clear(self) -> int:
        total = 0
        with self._session() as session:
            result = session.run(
                "MATCH ()-[r:CAUSAL_EDGE]->() "
                "DELETE r "
                "RETURN count(r) AS deleted"
            )
            record = result.single()
            if record:
                total += record["deleted"]

            result = session.run(
                "MATCH (n:CausalNode) "
                "DELETE n "
                "RETURN count(n) AS deleted"
            )
            record = result.single()
            if record:
                total += record["deleted"]

        logger.info("Cleared causal graph: %d nodes+edges removed", total)
        return total

    def export(self, file_path: str) -> None:
        nodes = self.get_all_nodes()
        edges = self.get_all_edges()
        data = {
            "nodes": [n.to_dict() for n in nodes],
            "edges": [e.to_dict() for e in edges],
            "exported_at": time.time(),
        }
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        logger.info(
            "Exported %d nodes and %d edges to %s",
            len(nodes), len(edges), file_path,
        )

    def import_(self, file_path: str) -> int:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        count = 0
        id_map: Dict[str, str] = {}
        for raw_node in data.get("nodes", []):
            name = raw_node.get("name", "")
            node_type = raw_node.get("type", "primitive")
            metadata = raw_node.get("metadata", {})
            existing = self.find_nodes_by_name(name)
            if existing:
                node_id = existing[0].id
            else:
                node_id = self.add_node(name, node_type, metadata)
                count += 1
            old_id = raw_node.get("id", "")
            if old_id:
                id_map[old_id] = node_id

        imported_edges = 0
        for raw_edge in data.get("edges", []):
            sid = id_map.get(raw_edge["source_id"], raw_edge["source_id"])
            tid = id_map.get(raw_edge["target_id"], raw_edge["target_id"])
            existing_edge = self.get_edge(sid, tid)
            if existing_edge is None:
                self.add_edge(
                    source_id=sid,
                    target_id=tid,
                    strength=float(raw_edge.get("strength", 0.0)),
                    p_value=float(raw_edge.get("p_value", 1.0)),
                    intervention_ids=list(
                        raw_edge.get("intervention_ids", [])
                    ),
                )
                imported_edges += 1

        total = count + imported_edges
        logger.info(
            "Imported %d nodes and %d edges from %s",
            count, imported_edges, file_path,
        )
        return total

    def _node_to_causal_node(self, node: Any) -> CausalNode:
        meta_str = node.get("metadata", "{}")
        try:
            metadata = json.loads(meta_str)
        except (json.JSONDecodeError, TypeError):
            metadata = {}
        return CausalNode(
            id=node.get("id", ""),
            name=node.get("name", ""),
            type=node.get("type", "primitive"),
            metadata=metadata,
            created_at=float(node.get("created_at", time.time())),
        )

    def _edge_to_causal_edge(self, rel: Any) -> CausalEdge:
        iids_str = rel.get("intervention_ids", "[]")
        try:
            intervention_ids = json.loads(iids_str)
        except (json.JSONDecodeError, TypeError):
            intervention_ids = []
        return CausalEdge(
            source_id=rel.get("source_id", ""),
            target_id=rel.get("target_id", ""),
            strength=float(rel.get("strength", 0.0)),
            p_value=float(rel.get("p_value", 1.0)),
            intervention_ids=intervention_ids,
            created_at=float(rel.get("created_at", time.time())),
        )

    # ------------------------------------------------------------------
    # Intervention-based causal discovery (Section 8 of harmony_v5v.md)
    # ------------------------------------------------------------------

    def do_intervention(
        self,
        intervention_id: str,
        source_name: str,
        target_name: str,
        outcomes_do_x0: List[int],
        outcomes_do_x1: List[int],
        use_bayesian: bool = True,
    ) -> Dict[str, Any]:
        """Record a do-operator intervention and update the causal edge.

        Simulates do(X=x) by recording outcomes under two conditions
        (X=x0 and X=x1), then computing the causal strength and p-value.

        Accumulates all outcomes across repeated interventions for the same
        (source, target) pair so that the Bayesian/Fisher test gains power
        as more data arrives (fix: each individual intervention had too few
        samples to ever reach p <= 0.05).

        Parameters
        ----------
        intervention_id : str
            Unique ID for this intervention.
        source_name : str
            Name of the cause node (X).
        target_name : str
            Name of the effect node (Y).
        outcomes_do_x0 : list of int
            Outcomes (0/1) observed when X is set to x0.
        outcomes_do_x1 : list of int
            Outcomes (0/1) observed when X is set to x1.
        use_bayesian : bool
            If True and bayes_mvs is available, use Bayesian test instead of
            Fisher's exact test. Bayesian works with smaller sample sizes.

        Returns
        -------
        dict with keys: strength, p_value, edge_added, ready

        Raises
        ------
        ValueError
            If either outcome list has fewer than the minimum observations.
        """
        if use_bayesian and _HAS_BAYES:
            min_per_group = 3
        else:
            min_per_group = _MIN_INTERVENTIONS_FOR_EDGE

        if len(outcomes_do_x0) < min_per_group or len(outcomes_do_x1) < min_per_group:
            raise ValueError(
                f"Each do-condition needs at least {min_per_group} observations, "
                f"got {len(outcomes_do_x0)} and {len(outcomes_do_x1)}"
            )

        total_obs = len(outcomes_do_x0) + len(outcomes_do_x1)

        if not (use_bayesian and _HAS_BAYES):
            if total_obs < _MIN_TOTAL_OBSERVATIONS:
                logger.debug(
                    "Insufficient total observations for do(%s): "
                    "got %d, need at least %d",
                    intervention_id, total_obs, _MIN_TOTAL_OBSERVATIONS,
                )
                return {
                    "strength": 0.0,
                    "p_value": 1.0,
                    "edge_added": False,
                    "ready": False,
                    "source_id": "",
                    "target_id": "",
                }

        source_id = self.get_or_create_node(source_name, "primitive")
        target_id = self.get_or_create_node(target_name, "primitive")

        # Accumulate outcomes across repeated interventions for the same edge.
        # Each individual intervention has too few samples for statistical power;
        # only by pooling all trials can we reliably detect causation.
        ck = self._cum_key(source_name, target_name)
        if ck in self._cumulative_outcomes:
            prev = self._cumulative_outcomes[ck]
            prev["x0"].extend(outcomes_do_x0)
            prev["x1"].extend(outcomes_do_x1)
            cum_x0 = prev["x0"]
            cum_x1 = prev["x1"]
        else:
            cum_x0 = list(outcomes_do_x0)
            cum_x1 = list(outcomes_do_x1)
            self._cumulative_outcomes[ck] = {"x0": cum_x0, "x1": cum_x1}

        if use_bayesian and _HAS_BAYES:
            p_value = self._compute_bayesian_p_value(cum_x0, cum_x1)
        else:
            p_value = self._compute_fisher_p_value(cum_x0, cum_x1)
        strength = self._compute_causal_strength(cum_x0, cum_x1)

        total_trials = len(cum_x0) + len(cum_x1)

        existing_edge = self.get_edge(source_id, target_id)
        base_iids: List[str] = [intervention_id]

        if existing_edge is not None:
            base_iids = list(
                set(existing_edge.intervention_ids + base_iids)
            )

        edge_added = False
        try:
            edge_added = self.add_edge(
                source_id=source_id,
                target_id=target_id,
                strength=strength,
                p_value=p_value,
                intervention_ids=base_iids,
                num_trials=total_trials,
            )
        except ValueError as exc:
            logger.warning(
                "Edge not added for do(%s): p=%.4f > %.2f (%s). "
                "Accumulated %d trials (cum_x0=%d cum_x1=%d) — need more data.",
                intervention_id, p_value, _P_VALUE_THRESHOLD, exc,
                total_trials, len(cum_x0), len(cum_x1),
            )

        return {
            "strength": strength,
            "p_value": p_value,
            "edge_added": edge_added,
            "ready": True,
            "source_id": source_id,
            "target_id": target_id,
            "cumulative_trials": total_trials,
        }

    @staticmethod
    def _compute_fisher_p_value(
        outcomes_x0: List[int],
        outcomes_x1: List[int],
    ) -> float:
        """Compute p-value using Fisher's exact test.

        Builds a 2x2 contingency table:
                        Y=0    Y=1
            do(X=x0)   a      b
            do(X=x1)   c      d

        Returns 1.0 if scipy is not available.
        """
        a = outcomes_x0.count(0)
        b = outcomes_x0.count(1)
        c = outcomes_x1.count(0)
        d = outcomes_x1.count(1)

        if fisher_exact is not None:
            _, p_value = fisher_exact([[a, b], [c, d]])
            return float(p_value)
        logger.warning(
            "scipy.stats.fisher_exact not available; "
            "falling back to heuristic p-value"
        )
        return _heuristic_p_value(a, b, c, d)

    @staticmethod
    def _compute_bayesian_p_value(outcomes_x0: List[int], outcomes_x1: List[int]) -> float:
        """Bayesian test for difference between two groups.

        Uses a Beta-Bernoulli model: P(p_x0 > p_x1 | data).
        Returns 1 - P(p_x0 > p_x1) as a pseudo p-value.
        """
        import numpy as np

        n_x0 = len(outcomes_x0)
        n_x1 = len(outcomes_x1)

        if n_x0 == 0 or n_x1 == 0:
            return 1.0

        a_x0 = sum(outcomes_x0) + 1  # Beta prior alpha=1
        b_x0 = n_x0 - sum(outcomes_x0) + 1  # Beta prior beta=1
        a_x1 = sum(outcomes_x1) + 1
        b_x1 = n_x1 - sum(outcomes_x1) + 1

        # Monte Carlo: sample from Beta posteriors
        n_samples = 10000
        rng = np.random.default_rng()
        samples_x0 = rng.beta(a_x0, b_x0, n_samples)
        samples_x1 = rng.beta(a_x1, b_x1, n_samples)

        # P(p_x0 > p_x1)
        prob_x0_gt_x1 = np.mean(samples_x0 > samples_x1)

        # Return two-tailed pseudo p-value
        return 2 * min(prob_x0_gt_x1, 1 - prob_x0_gt_x1)

    @staticmethod
    def _compute_causal_strength(
        outcomes_x0: List[int],
        outcomes_x1: List[int],
    ) -> float:
        """strength(X->Y) = |P(Y=1|do(X=x1)) - P(Y=1|do(X=x0))|."""
        p_x0 = sum(outcomes_x0) / max(len(outcomes_x0), 1)
        p_x1 = sum(outcomes_x1) / max(len(outcomes_x1), 1)
        return abs(p_x1 - p_x0)


def _heuristic_p_value(a: int, b: int, c: int, d: int) -> float:
    """Heuristic p-value approximation when scipy is unavailable.

    Uses the chi-squared-like formula:
        chi2 = (ad - bc)^2 * N / ((a+b)(c+d)(a+c)(b+d))
    where N = a+b+c+d, then converts to approximate p-value.
    """
    n = a + b + c + d
    if n == 0:
        return 1.0
    denom = (a + b) * (c + d) * (a + c) * (b + d)
    if denom == 0:
        return 1.0
    chi2 = float((a * d - b * c) ** 2) * n / float(denom)
    from math import exp
    return exp(-chi2 / 2.0)
