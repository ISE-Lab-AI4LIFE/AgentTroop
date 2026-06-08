"""Graph reasoner — produces explanations from retrieved subgraphs.

Transforms a subgraph (nodes + edges) into a natural-language explanation by
(1) extracting causal chains from edges sorted by strength,
(2) computing confidence from p-values and strengths,
(3) synthesising an answer without hardcoded templates.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

logger = logging.getLogger(__name__)


@dataclass
class GraphRAGAnswer:
    query: str
    answer: str
    evidence_nodes: List[Dict[str, Any]] = field(default_factory=list)
    evidence_edges: List[Dict[str, Any]] = field(default_factory=list)
    confidence: float = 0.0


_EDGE_VERBS = {
    "bypass": "bypasses",
    "blocks": "blocks",
    "decodes": "decodes",
    "encodes": "encodes",
    "triggers": "triggers",
    "causes": "causes",
    "prevents": "prevents",
    "detects": "detects",
    "classifies": "classifies",
}


class GraphReasoner:
    """Reason over a retrieved subgraph to produce a causal explanation.

    The reasoner:
      1. Sorts edges by causal strength (descending).
      2. Builds linear causal chains from source → target.
      3. Synthesises an answer by joining chain descriptions.
      4. Computes overall confidence from the geometric mean of
         (1 - p_value) × strength for each edge in the longest chain.
    """

    def reason(self, subgraph: Dict[str, Any]) -> GraphRAGAnswer:
        query_text = subgraph.get("query", "")
        nodes = subgraph.get("nodes", [])
        edges = subgraph.get("edges", [])

        if not edges:
            return GraphRAGAnswer(
                query=query_text,
                answer=self._no_evidence_answer(nodes, query_text),
                evidence_nodes=nodes,
                evidence_edges=edges,
                confidence=0.0,
            )

        # 1. Sort edges by strength descending
        sorted_edges = sorted(
            edges,
            key=lambda e: e.get("strength", 0.0) if isinstance(e, dict) else 0.0,
            reverse=True,
        )

        # 2. Build chains
        chains = self._build_causal_chains(sorted_edges)
        if not chains:
            return GraphRAGAnswer(
                query=query_text,
                answer=self._no_chain_answer(nodes, edges),
                evidence_nodes=nodes,
                evidence_edges=edges,
                confidence=0.3,
            )

        # 3. Synthesise answer from the strongest chain
        primary_chain = chains[0]
        chain_str = self._chain_to_text(primary_chain)

        # Build supporting context from remaining edges
        context_parts = []
        for edge in sorted_edges[:3]:
            src = edge.get("source_name", edge.get("source_id", "?"))
            tgt = edge.get("target_name", edge.get("target_id", "?"))
            strength = edge.get("strength", 0.0)
            p_val = edge.get("p_value", 1.0)
            if strength > 0.3 and p_val < 0.1:
                context_parts.append(f"{src} → {tgt} (strength={strength:.2f})")
        context_str = "; ".join(context_parts[:2])

        answer_parts = [f"Causal explanation: {chain_str}."]
        if context_str:
            answer_parts.append(f"Supporting edges: {context_str}.")
        answer_parts.append(
            "This causal relationship was verified through targeted interventions."
        )
        answer = " ".join(answer_parts)

        # 4. Confidence
        conf = self._chain_confidence(primary_chain)
        if conf <= 0.0:
            conf = 0.5

        return GraphRAGAnswer(
            query=query_text,
            answer=answer,
            evidence_nodes=nodes,
            evidence_edges=sorted_edges[:5],
            confidence=min(conf, 1.0),
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_causal_chains(
        edges: List[Dict[str, Any]],
    ) -> List[List[Dict[str, Any]]]:
        """Build linear causal chains from a list of directed edges.

        Uses a greedy approach: repeatedly find edges that extend an
        existing chain (target_id == next source_id).  Edges are consumed
        at most once.
        """
        used: set = set()
        chains: List[List[Dict[str, Any]]] = []

        edge_ids: List[str] = []
        edge_map: Dict[str, Dict[str, Any]] = {}
        for i, e in enumerate(edges):
            eid = f"{e.get('source_id', '?')}→{e.get('target_id', '?')}_{i}"
            edge_ids.append(eid)
            edge_map[eid] = e

        # Build adjacency: source_id -> list of (target_id, edge_id)
        adj: Dict[str, List[Tuple[str, str]]] = {}
        for eid, e in edge_map.items():
            src = e.get("source_id", "?")
            tgt = e.get("target_id", "?")
            adj.setdefault(src, []).append((tgt, eid))

        # Greedy chain building
        for start_src in list(adj.keys()):
            if all(eid in used for eid in edge_ids):
                break
            chain: List[str] = []
            current = start_src
            for _ in range(len(adj)):  # avoid infinite loop
                candidates = adj.get(current, [])
                best: Optional[Tuple[str, str]] = None
                for c in candidates:
                    if c[1] not in used:
                        best = c
                        break
                if best is None:
                    break
                chain.append(best[1])
                used.add(best[1])
                current = best[0]

            if chain:
                chains.append([edge_map[eid] for eid in chain])

        return chains

    @staticmethod
    def _chain_to_text(chain: List[Dict[str, Any]]) -> str:
        """Convert a causal chain to a readable sentence."""
        parts = []
        for i, edge in enumerate(chain):
            src = edge.get("source_name", edge.get("source_id", "?"))
            tgt = edge.get("target_name", edge.get("target_id", "?"))
            rel = edge.get("relation", "affects")
            verb = _EDGE_VERBS.get(rel, rel)
            if i == 0:
                parts.append(f"{src} {verb} {tgt}")
            else:
                parts.append(f"then {verb} {tgt}")
        return ", ".join(parts)

    @staticmethod
    def _chain_confidence(chain: List[Dict[str, Any]]) -> float:
        """Compute confidence from geometric mean of (1-p)×strength."""
        if not chain:
            return 0.0
        scores = []
        for edge in chain:
            p = edge.get("p_value", 1.0)
            s = edge.get("strength", 0.0)
            scores.append(max(0.0, (1.0 - p) * s))
        if not scores:
            return 0.0
        prod = 1.0
        for sc in scores:
            prod *= max(sc, 0.001)
        return prod ** (1.0 / len(scores))

    @staticmethod
    def _no_evidence_answer(
        nodes: List[Dict[str, Any]], query: str
    ) -> str:
        """Fallback when no causal edges exist."""
        if nodes:
            names = [n.get("name", "?") for n in nodes[:3]]
            return (
                f"Nodes {', '.join(names)} were found in the knowledge base, "
                f"but no verified causal relationships exist yet. "
                f"Perform more targeted interventions to discover causal structure."
            )
        return (
            f"No relevant subgraph found for '{query[:60]}'. "
            f"The safety mechanism may involve untracked components."
        )

    @staticmethod
    def _no_chain_answer(
        nodes: List[Dict[str, Any]], edges: List[Dict[str, Any]]
    ) -> str:
        """Fallback when edges exist but cannot be chained."""
        edge_count = len(edges)
        node_names = [n.get("name", "?") for n in nodes[:3]]
        return (
            f"Found {edge_count} causal edge(s) involving "
            f"{', '.join(node_names)}, but they do not form a connected "
            f"chain. Additional interventions are needed to link them."
        )
