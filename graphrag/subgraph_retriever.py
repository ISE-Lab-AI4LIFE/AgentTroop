"""Subgraph retriever — queries CausalGraph and DefenseProgramStore."""

import logging
from typing import Any, Dict, List

from graphrag.query_parser import GraphRAGQuery

logger = logging.getLogger(__name__)


class SubgraphRetriever:
    def __init__(self, causal_graph: Any, defense_store: Any) -> None:
        self._causal = causal_graph
        self._defense = defense_store

    def retrieve(self, query: GraphRAGQuery) -> Dict[str, Any]:
        nodes: List[Dict[str, Any]] = []
        edges: List[Dict[str, Any]] = []
        seen_ids: set = set()

        for entity in query.target_entities:
            try:
                cnodes = self._causal.find_nodes_by_name(entity)
                for n in cnodes:
                    nid = n.id
                    if nid not in seen_ids:
                        seen_ids.add(nid)
                        nodes.append(n.to_dict() if hasattr(n, "to_dict") else {"id": nid, "name": entity})
                    causes = self._causal.find_causes(target_id=nid)
                    for ce, cnode in causes:
                        eid = ce.source_id + "->" + ce.target_id
                        edges.append({
                            "source_id": ce.source_id,
                            "target_id": ce.target_id,
                            "source_name": cnode.name,
                            "target_name": n.name,
                            "strength": ce.strength,
                            "p_value": ce.p_value,
                        })
                    effects = self._causal.find_effects(source_id=nid)
                    for ce, enode in effects:
                        eid = ce.source_id + "->" + ce.target_id
                        edges.append({
                            "source_id": ce.source_id,
                            "target_id": ce.target_id,
                            "source_name": n.name,
                            "target_name": enode.name,
                            "strength": ce.strength,
                            "p_value": ce.p_value,
                        })
            except Exception as exc:
                logger.debug("Failed to retrieve causal node '%s': %s", entity, exc)

            try:
                programs = self._defense.get_programs_by_name(entity)
                for prog in programs:
                    pid = getattr(prog, "id", str(prog))
                    if pid not in seen_ids:
                        seen_ids.add(pid)
                        nodes.append({
                            "id": pid,
                            "name": entity,
                            "type": "program",
                            "source": "defense_store",
                        })
            except Exception as exc:
                logger.debug("Failed to retrieve defense program '%s': %s", entity, exc)

        return {"nodes": nodes, "edges": edges, "query": query.raw_text}
